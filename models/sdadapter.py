import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import timm
from backbone.adapter import SDAdapter_ViT_timm
import torch.distributed as dist

import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.data_manager = data_manager
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # ---- Classifier Alignment (CA) ----
        ca_epochs = self.args.get("ca_epochs", 0)
        if ca_epochs > 0:
            self._compute_class_mean(data_manager)
            if self._cur_task > 0:
                self._stage2_compact_classifier()
                save_dir = self.args['filepath']
                self._network.save_fc(save_dir, self._cur_task)
                logging.info("Re-saved fc after CA stage2 for task {}".format(self._cur_task))

    def update_network(self, index=True):
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

        adapter_rank = self.args.get('adapter_rank', 64)
        per_layer_scaling = self.args.get('per_layer_scaling', False)
        model = SDAdapter_ViT_timm(
            vit_model=model.eval(), r=adapter_rank, num_classes=10,
            index=index, increment=self.args['increment'],
            filepath=self.args['filepath'],
            cur_task_index=self._cur_task,
            per_layer_scaling=per_layer_scaling
        )
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._network.to(self._device)

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_name = self.args['filepath']

        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_adapter_parameters(save_name, self._cur_task)
            self._network.module.save_fc(save_name, self._cur_task)
        else:
            self._network.backbone.save_adapter_parameters(save_name, self._cur_task)
            self._network.save_fc(save_name, self._cur_task)
        # Save scaling factors and a single-file checkpoint for this task
        self._save_task_checkpoint()

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                self.args["lrate"],
                betas=(0.9, 0.999)
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args['min_lr'])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _save_task_checkpoint(self):
        """Save per-task checkpoint after training finishes."""
        save_dir = self.args.get('filepath', './')
        os.makedirs(save_dir, exist_ok=True)

        net = self._network.module if len(self._multiple_gpus) > 1 else self._network

        # Save scaling matrix
        if hasattr(net, 'backbone') and hasattr(net.backbone, 'save_wrap_param'):
            try:
                net.backbone.save_wrap_param(save_dir)
            except Exception as e:
                logging.warning(f"Failed to save scaling matrix: {e}")

        # Pack a unified checkpoint
        ckpt_path = os.path.join(save_dir, f"ckpt_task_{self._cur_task}.pt")
        try:
            backbone_state = net.backbone.state_dict() if hasattr(net, 'backbone') else {}
            fc_state = net.fc.state_dict() if hasattr(net, 'fc') else {}
            ckpt = {
                'task': self._cur_task,
                'args': self.args,
                'backbone_state': backbone_state,
                'fc_state': fc_state,
            }
            torch.save(ckpt, ckpt_path)
            logging.info(f"Saved task checkpoint to: {ckpt_path}")
        except Exception as e:
            logging.warning(f"Failed to save task checkpoint: {e}")

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                logits = logits['logits']

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], fake_targets
                )

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    # ===================== Classifier Alignment (CA) =====================

    def _compute_class_mean(self, data_manager):
        """Compute class-conditional mean and covariance for new classes only."""
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        feat_dim = net.feature_dim

        if not hasattr(self, '_ca_class_means') or self._ca_class_means is None:
            self._ca_class_means = np.zeros((self._total_classes, feat_dim))
            self._ca_class_covs = np.zeros((self._total_classes, feat_dim, feat_dim))
        else:
            if self._ca_class_means.shape[0] < self._total_classes:
                old_means = self._ca_class_means
                old_covs = self._ca_class_covs
                self._ca_class_means = np.zeros((self._total_classes, feat_dim))
                self._ca_class_covs = np.zeros((self._total_classes, feat_dim, feat_dim))
                self._ca_class_means[:old_means.shape[0]] = old_means
                self._ca_class_covs[:old_covs.shape[0]] = old_covs

        logging.info(
            "Computing class statistics for new classes [{}, {}) "
            "(old classes [0, {}) use cached statistics) ...".format(
                self._known_classes, self._total_classes, self._known_classes
            )
        )
        for class_idx in range(self._known_classes, self._total_classes):
            _, _, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train", mode="test", ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.args["batch_size"],
                shuffle=False, num_workers=num_workers,
            )
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = np.mean(vectors, axis=0)
            self._ca_class_means[class_idx] = class_mean

            vectors_t = torch.tensor(vectors)
            if vectors_t.shape[0] > 1:
                cov = torch.cov(vectors_t.T)
            else:
                cov = torch.zeros(feat_dim, feat_dim)
            cov += 1e-4 * torch.eye(feat_dim)

            try:
                torch.linalg.cholesky(cov)
            except RuntimeError:
                logging.warning("Cov not PD for class {}, adding extra jitter".format(class_idx))
                cov += 1e-2 * torch.eye(feat_dim)

            self._ca_class_covs[class_idx] = cov.numpy()

        logging.info("Class statistics computed.")

    def _stage2_compact_classifier(self):
        """CA stage 2: retrain the classifier using sampled pseudo-features."""
        ca_epochs = self.args.get("ca_epochs", 5)
        ca_lr = self.args.get("ca_lr", 0.01)
        ca_num_samples = self.args.get("ca_num_samples", 256)
        logit_norm = self.args.get("logit_norm", None)

        task_sizes = []
        for t in range(self._cur_task + 1):
            ts = self.data_manager.get_task_size(t)
            task_sizes.append(ts)

        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        for p in net.fc.parameters():
            p.requires_grad = True

        optimizer = optim.SGD(
            net.fc.parameters(), lr=ca_lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ca_epochs)

        logging.info(
            "CA stage2: {} epochs, lr={}, samples/cls={}, logit_norm={}".format(
                ca_epochs, ca_lr, ca_num_samples, logit_norm
            )
        )

        for epoch in range(ca_epochs):
            sampled_data_list = []
            sampled_label_list = []

            for c_id in range(self._total_classes):
                _acc = 0
                t_id = 0
                for _t, _ts in enumerate(task_sizes):
                    _acc += _ts
                    if c_id < _acc:
                        t_id = _t
                        break
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = torch.tensor(
                    self._ca_class_means[c_id], dtype=torch.float64
                ) * (0.9 + decay)
                cls_cov = torch.tensor(
                    self._ca_class_covs[c_id], dtype=torch.float64
                )

                m = torch.distributions.MultivariateNormal(cls_mean, cls_cov)
                sampled = m.sample(sample_shape=(ca_num_samples,))
                sampled_data_list.append(sampled)
                sampled_label_list.append(
                    torch.full((ca_num_samples,), c_id, dtype=torch.long)
                )

            sampled_data = torch.cat(sampled_data_list, dim=0).float().to(self._device)
            sampled_label = torch.cat(sampled_label_list, dim=0).to(self._device)

            perm = torch.randperm(sampled_data.size(0))
            sampled_data = sampled_data[perm]
            sampled_label = sampled_label[perm]

            batch_size = min(256, sampled_data.size(0))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, sampled_data.size(0), batch_size):
                batch_feat = sampled_data[i : i + batch_size]
                batch_label = sampled_label[i : i + batch_size]

                logits = net(batch_feat, fc_only=True)["logits"]

                if logit_norm is not None:
                    norms = []
                    idx = 0
                    for ts in task_sizes:
                        task_logits = logits[:, idx : idx + ts]
                        norms.append(
                            torch.norm(task_logits, p=2, dim=1, keepdim=True) + 1e-7
                        )
                        idx += ts
                    norms = torch.cat(norms, dim=1)
                    avg_norm = norms.mean(dim=1, keepdim=True)
                    logits = logits / (avg_norm + 1e-7) / logit_norm

                loss = F.cross_entropy(logits, batch_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            if (epoch + 1) % max(1, ca_epochs // 5) == 0 or epoch == 0:
                logging.info(
                    "  CA epoch {}/{} loss={:.4f}".format(
                        epoch + 1, ca_epochs, epoch_loss / max(n_batches, 1)
                    )
                )

        for p in net.parameters():
            p.requires_grad = True

        logging.info("CA stage2 completed for task {}".format(self._cur_task))

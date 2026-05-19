
import torch
import torch.nn as nn


class ConstantEncoder(nn.Module):
    def __init__(self, args=None, in_dim=None, hidden_dim=None, out_dim=None):
        super().__init__()

        if args is not None:
            self.in_dim = getattr(args, 'static_dim', in_dim or 10)
            self.hidden_dim = getattr(args, 'static_hidden_dim', hidden_dim or 64)
            self.out_dim = getattr(args, 'static_emb_dim', out_dim or 32)
        else:
            self.in_dim = in_dim or 10
            self.hidden_dim = hidden_dim or 64
            self.out_dim = out_dim or 32

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )

    def forward(self, s):
        return self.net(s)


class TwoStageModulator(nn.Module):
    def __init__(
        self,
        ts_hidden_dim,
        static_emb_dim,
        cluster_emb_dim,
        coarse_rank=8,
    ):
        super().__init__()
        self.mod_obj_dim = ts_hidden_dim + static_emb_dim + cluster_emb_dim

        self.static_film = nn.Linear(static_emb_dim, self.mod_obj_dim * 2)
        self.cluster_film = nn.Linear(cluster_emb_dim, self.mod_obj_dim * 2)
        self.time_film = nn.Linear(self.mod_obj_dim, self.mod_obj_dim * 2)

        self.coarse_router = nn.Linear(static_emb_dim + cluster_emb_dim, coarse_rank)
        self.coarse_basis = nn.Parameter(torch.randn(coarse_rank, self.mod_obj_dim) * 0.02)
        self.coarse_gate = nn.Sequential(
            nn.Linear(static_emb_dim + cluster_emb_dim, self.mod_obj_dim),
            nn.Sigmoid(),
        )
        self.fine_gate = nn.Sequential(
            nn.Linear(self.mod_obj_dim, self.mod_obj_dim),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, fused_base, static_emb, cluster_emb):
        # Stage 1: static + cluster coarse shift.
        scale_s, shift_s = self.static_film(static_emb).chunk(2, dim=-1)
        scale_c, shift_c = self.cluster_film(cluster_emb).chunk(2, dim=-1)
        alpha_w = torch.sigmoid(self.alpha)
        scale_sc = alpha_w * scale_s + (1 - alpha_w) * scale_c
        shift_sc = alpha_w * shift_s + (1 - alpha_w) * shift_c
        fused_sc_film = fused_base * (1.0 + scale_sc) + shift_sc

        coarse_ctx = torch.cat([static_emb, cluster_emb], dim=-1)
        coarse_coeff = self.coarse_router(coarse_ctx)
        coarse_dir = coarse_coeff @ self.coarse_basis
        coarse_gate = self.coarse_gate(coarse_ctx)
        coarse_delta = coarse_gate * (fused_sc_film - fused_base) + (1.0 - coarse_gate) * coarse_dir
        fused_sc = fused_base + coarse_delta

        # Stage 2: temporal fine-grained shift.
        scale_t, shift_t = self.time_film(fused_sc).chunk(2, dim=-1)
        fused_t = fused_sc * (1.0 + scale_t) + shift_t
        fine_delta_raw = fused_t - fused_sc
        fine_gate = self.fine_gate(fused_sc) * (1.0 - coarse_gate)
        fine_delta = fine_gate * fine_delta_raw
        fused_mod = fused_sc + fine_delta

        return {
            'fused_mod': fused_mod,
            'coarse_delta': coarse_delta,
            'fine_delta': fine_delta,
        }

class CGDPM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_model = getattr(args, 'time_model', 'Transformer')
        self.pred_len = getattr(args, 'pred_len', 1)
        self.ts_dim = getattr(args, 'ts_dim', 1)
        self.static_dim = getattr(args, 'static_dim', 36)
        self.num_clusters = getattr(args, 'num_clusters', 8)
        self.num_classes = getattr(args, 'num_classes', 4)
        self.static_emb_dim = getattr(args, 'static_emb_dim', 32)
        self.cluster_emb_dim = getattr(args, 'cluster_emb_dim', 16)
        self.ts_hidden_dim = getattr(args, 'ts_hidden_dim', 64)
        self.seq_len = getattr(args, 'seq_len', 251)
        self.label_len = getattr(args, 'label_len', 48)
        self.moving_avg = getattr(args, 'moving_avg', 25)

        self.model_dict = self._build_model_dict()

        self.static_encoder = ConstantEncoder(args=args, in_dim=self.static_dim, out_dim=self.static_emb_dim)
        self.cluster_embedding = nn.Embedding(self.num_clusters, self.cluster_emb_dim)
        self.cluster_head = nn.Linear(self.static_emb_dim, self.num_clusters)
        self.time_encoder = self._init_time_encoder(args)

        # Adapt to different time-encoder output shapes.
        self.temporal_projection = nn.LazyLinear(self.ts_hidden_dim)

        self.coarse_rank = getattr(args, 'coarse_rank', 8)
        self.modulator = TwoStageModulator(
            ts_hidden_dim=self.ts_hidden_dim,
            static_emb_dim=self.static_emb_dim,
            cluster_emb_dim=self.cluster_emb_dim,
            coarse_rank=self.coarse_rank,
        )

        self.fused_dim = self.modulator.mod_obj_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

        self.clustering_history = {
            'cluster_sizes': [],
            'cluster_probs_mean': []
        }

    def _build_model_dict(self):
        model_dict = {}
        base_models = {
            'Transformer': 'models.encoders.Transformer',
            'Autoformer': 'models.encoders.Autoformer',
            'TimesNet': 'models.encoders.TimesNet',
            'PatchTST': 'models.encoders.PatchTST',
            'Crossformer': 'models.encoders.Crossformer',
            'FEDformer': 'models.encoders.FEDformer',
            'Informer': 'models.encoders.Informer',
            'DLinear': 'models.encoders.DLinear',
            'TimeFilter': 'models.encoders.TimeFilter',
            'TimeMixer': 'models.encoders.TimeMixer',
        }
        for model_name, module_path in base_models.items():
            try:
                import importlib
                module = importlib.import_module(module_path)
                model_class = getattr(module, 'Model', None)
                if model_class:
                    model_dict[model_name] = model_class
            except ImportError:
                pass
        return model_dict

    def _init_time_encoder(self, args):
        if self.time_model not in self.model_dict:
            raise ValueError(f"Unsupported time model: {self.time_model}. Available: {list(self.model_dict.keys())}")

        ts_dim = self.ts_dim
        num_classes = self.num_classes

        class TimeModelConfig:
            def __init__(self):
                self.task_name = getattr(args, 'task_name', 'classification')
                self.enc_in = ts_dim
                self.dec_in = ts_dim
                self.c_out = ts_dim
                self.ts_dim = ts_dim
                self.num_class = num_classes
                self.seq_len = getattr(args, 'seq_len', 251)
                self.label_len = getattr(args, 'label_len', 48)
                self.pred_len = getattr(args, 'pred_len', 1)
                self.d_model = getattr(args, 'd_model', 512)
                self.n_heads = getattr(args, 'n_heads', 8)
                self.d_ff = getattr(args, 'd_ff', 2048)
                self.e_layers = getattr(args, 'e_layers', 2)
                self.d_layers = getattr(args, 'd_layers', 1)
                self.embed = getattr(args, 'embed', 'timeF')
                self.freq = getattr(args, 'freq', 'd') if hasattr(args, 'freq') else 'd'
                self.dropout = getattr(args, 'dropout', 0.1)
                self.activation = getattr(args, 'activation', 'gelu')
                self.factor = getattr(args, 'factor', 1)
                self.moving_avg = getattr(args, 'moving_avg', 25)
                self.distil = getattr(args, 'distil', True)
                self.top_k = getattr(args, 'top_k', 5)
                self.num_kernels = getattr(args, 'num_kernels', 6)
                self.expand = getattr(args, 'expand', 2)
                self.d_conv = getattr(args, 'd_conv', 4)
                self.channel_independence = getattr(args, 'channel_independence', 1)
                self.decomp_method = getattr(args, 'decomp_method', 'moving_avg')
                self.use_norm = getattr(args, 'use_norm', 1)
                self.down_sampling_layers = getattr(args, 'down_sampling_layers', 0)
                self.down_sampling_window = getattr(args, 'down_sampling_window', 1)
                self.down_sampling_method = getattr(args, 'down_sampling_method', None)
                self.seg_len = getattr(args, 'seg_len', 96)
                self.revin = getattr(args, 'revin', False)
                # TimeFilter options
                self.patch_len = getattr(args, 'patch_len', 16)
                self.stride = getattr(args, 'stride', 16)
                self.alpha = getattr(args, 'alpha', 0.1)
                self.top_p = getattr(args, 'top_p', 0.5)
                self.pos = getattr(args, 'pos', True)

        return self.model_dict[self.time_model](TimeModelConfig())

    def forward(self, x, s, x_mark=None):
        static_emb = self.static_encoder(s)
        cluster_logits = self.cluster_head(static_emb)
        cluster_probs = torch.softmax(cluster_logits, dim=-1)
        c = cluster_probs @ self.cluster_embedding.weight  # [B, cluster_emb_dim]

        B, T = x.shape[0], x.shape[1]
        if x_mark is None:
            x_mark = torch.ones(B, T, device=x.device, dtype=torch.float32)
        elif x_mark.ndim == 3:
            x_mark = torch.ones(B, T, device=x.device, dtype=torch.float32)

        z_raw = self.time_encoder(x, x_mark, None, None)
        if z_raw.ndim == 3:
            z_raw = z_raw.reshape(B, -1)
        z_ts = self.temporal_projection(z_raw)  # [B, ts_hidden_dim]
        fused_base = torch.cat([z_ts, static_emb, c], dim=-1)

        mod_out = self.modulator(fused_base=fused_base, static_emb=static_emb, cluster_emb=c)
        fused_mod = mod_out['fused_mod']
        coarse_delta = mod_out['coarse_delta']
        fine_delta = mod_out['fine_delta']

        logits = self.classifier(fused_mod)
        mod_delta = fused_mod - fused_base
        return logits, cluster_probs, {
            "z_mod": fused_mod,
            "z_base": fused_base,
            "mod_delta": mod_delta,
            "coarse_delta": coarse_delta,
            "fine_delta": fine_delta,
        }

    def compute_latent_orthogonality_loss(self, coarse_delta, fine_delta):
        coarse_norm = torch.norm(coarse_delta, dim=-1) + 1e-8
        fine_norm = torch.norm(fine_delta, dim=-1) + 1e-8
        cosine = torch.sum(coarse_delta * fine_delta, dim=-1) / (coarse_norm * fine_norm)
        return torch.mean(cosine ** 2)

    def compute_cluster_entropy_loss(self, cluster_probs):
        entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=-1)
        return -entropy.mean()

    def record_cluster_stats(self, cluster_probs):
        with torch.no_grad():
            cluster_ids = cluster_probs.argmax(dim=-1)
            cluster_sizes = torch.bincount(cluster_ids, minlength=self.num_clusters).cpu().numpy()
            self.clustering_history['cluster_sizes'].append(cluster_sizes)
            self.clustering_history['cluster_probs_mean'].append(cluster_probs.mean(dim=0).cpu().numpy())

    def get_clustering_stats(self):
        return self.clustering_history

    def clear_clustering_history(self):
        self.clustering_history = {
            'cluster_sizes': [],
            'cluster_probs_mean': []
        }

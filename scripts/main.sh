# 1) Transformer
python main.py --model_id CGDPM_Transformer --time_model Transformer --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des Transformer_run

# 2) Autoformer
python main.py --model_id CGDPM_Autoformer --time_model Autoformer --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des Autoformer_run

# 3) TimesNet
python main.py --model_id CGDPM_TimesNet --time_model TimesNet --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des TimesNet_run

# 4) PatchTST
python main.py --model_id CGDPM_PatchTST --time_model PatchTST --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des PatchTST_run

# 5) Crossformer
python main.py --model_id CGDPM_Crossformer --time_model Crossformer --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des Crossformer_run

# 6) FEDformer
python main.py --model_id CGDPM_FEDformer --time_model FEDformer --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des FEDformer_run

# 7) Informer
python main.py --model_id CGDPM_Informer --time_model Informer --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des Informer_run

# 8) DLinear
python main.py --model_id CGDPM_DLinear --time_model DLinear --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des DLinear_run

# 9) TimeFilter
python main.py --model_id CGDPM_TimeFilter --time_model TimeFilter --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des TimeFilter_run

# 10) TimeMixer
python main.py --model_id CGDPM_TimeMixer --time_model TimeMixer --data UEA --data_path ./data --uea_file sample_data.ts --model CGDPM --num_epochs 100 --batch_size 32 --lr 0.001 --num_clusters 8 --num_classes 4 --static_dim 36 --static_emb_dim 32 --cluster_emb_dim 16 --ts_hidden_dim 64 --coarse_rank 8 --label_len 48 --norm True --cluster_dispersion_weight 0.01 --latent_sep_weight 0.01 --output_dir ./outputs --des TimeMixer_run
def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('DAPS_ml1m'):
        args.mode = 'train'
        args.experiment_description ='ml1m_pop_DAPS'
        args.device_idx = '7'

        args.test_path = './experiments/ml1m_pop_DAPS'

        args.dataset_code = 'ml-1m'
        args.min_rating = 0
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'gru+sas'
        batch = 256
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.max_len = 200
        args.sample_num = 40
        
        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'dansrec'
        args.device = 'cuda'
        args.num_gpu = 1
        args.optimizer = 'Adam'
        args.lr = 0.002
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.weight_decay = 0.00000
        args.best_metric = 'NDCG@10'
        args.model_code = 'dansrec'
        args.model_init_seed = 0
        args.sas_n_layers = 4
        args.sas_n_heads = 2
        args.sas_hidden_size = 100
        args.sas_inner_size = 100*4
        args.sas_attn_dropout_prob = 0.3
        args.sas_hidden_dropout_prob = 0.5
        args.sas_hidden_act = 'gelu'
        args.sas_layer_norm_eps = 1e-12
        args.sas_initializer_range = 0.02

    elif args.template.startswith('DAPS_ml20m'):
        args.mode = 'train'
        args.experiment_description ='ml20m_pop_DAPS'
        args.device_idx = '7'
        args.test_path = './experiments/ml20m_pop_DAPS'


        args.dataset_code = 'ml-20m'
        args.min_rating = 0
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'gru+sas'
        batch = 256
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.max_len = 200
        args.sample_num = 20

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'dansrec'
        args.device = 'cuda'
        args.num_gpu = 1
        args.optimizer = 'Adam'
        args.lr = 0.002
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.weight_decay = 0.00000
        args.best_metric = 'NDCG@10'
        args.model_code = 'dansrec'
        args.model_init_seed = 0
        args.sas_n_layers = 3
        args.sas_n_heads = 2
        args.sas_hidden_size = 128
        args.sas_inner_size = 128*4
        args.sas_attn_dropout_prob = 0.1
        args.sas_hidden_dropout_prob = 0.2
        args.sas_hidden_act = 'gelu'
        args.sas_layer_norm_eps = 1e-12
        args.sas_initializer_range = 0.02
    
    elif args.template.startswith('DAPS_beauty'):
        args.mode = 'train'
        args.experiment_description ='beauty_pop_DAPS'
        args.device_idx = '5'
        args.test_path = './experiments/beauty_pop_DAPS'

        args.dataset_code = 'Beauty'
        args.min_rating = 0
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'
        args.sample_num = 20

        args.dataloader_code = 'gru+sas'
        batch = 256
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.max_len = 50
        args.weight_decay = 5e-5

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'dansrec'
        args.device = 'cuda'
        args.num_gpu = 1
        args.optimizer = 'Adam'
        args.lr = 0.002
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'
        args.model_code = 'dansrec'
        args.model_init_seed = 0
        args.sas_n_layers = 4
        args.sas_n_heads = 2
        args.sas_hidden_size = 64
        args.sas_inner_size = 64*4
        args.sas_attn_dropout_prob = 0.4
        args.sas_hidden_dropout_prob = 0.5
        args.sas_hidden_act = 'gelu'
        args.sas_layer_norm_eps = 1e-12
        args.sas_initializer_range = 0.02

    elif args.template.startswith('DAPS_steam'):
        args.mode = 'train'
        args.experiment_description ='steam_pop_DAPS'
        args.device_idx = '0'
        args.test_path = './experiments/steam_pop_DAPS'


        args.dataset_code = 'Steam'
        args.min_rating = 0
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'gru+sas'
        batch = 256
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.max_len = 50
        args.sample_num = 25

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'dansrec'
        args.device = 'cuda'
        args.num_gpu = 1
        args.optimizer = 'Adam'
        args.lr = 0.002
        args.enable_lr_schedule = False
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 60
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.weight_decay = 0.00000
        args.best_metric = 'NDCG@10'
        args.model_code = 'dansrec'
        args.model_init_seed = 0
        args.sas_n_layers = 3
        args.sas_n_heads = 2
        args.sas_hidden_size = 128
        args.sas_inner_size = 128*4
        args.sas_attn_dropout_prob = 0.4
        args.sas_hidden_dropout_prob = 0.5
        args.sas_hidden_act = 'gelu'
        args.sas_layer_norm_eps = 1e-12
        args.sas_initializer_range = 0.02
    

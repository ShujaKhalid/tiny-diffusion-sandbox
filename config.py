class Config(object):
    def __init__(self):
        # Extract configuration parameters here
        self.cfg = self.__dict__.update({
            "seed": 0,

            # data
            "csv_file": "assets/simple/cat.csv",
            "dataset_size": 10000,

            # opt_params
            "epochs": 1000,
            "batch_size": 32,
            "lr": 2.5e-3,

            # model_params
            "model": "MLP",  # denoiser
            "input_dim": 3,  # FIXED
            "hidden_dim": 64,
            "output_dim": 2,
            "hidden_layers": 2,

            # output_params
            "log_dir": "./logs/",

            # noise_scheduler
            "beta_start": 1e-5,
            "beta_end": 1e-2,
            "timesteps": 50,
        })


    # def get_config(self):
    #     return self.__dict__.update(self.cfg)

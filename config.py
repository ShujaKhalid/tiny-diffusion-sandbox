class Config(object):
    def __init__(self):
        # Extract configuration parameters here
        self.cfg = self.__dict__.update({
            "seed": 0,

            # data
            "csv_file": "assets/simple/cat.csv",
            "dataset_size": 200,

            # opt_params
            "epochs": 1000,
            "batch_size": 1024,
            "lr": 1e-3,

            # model_params
            "model": "MLP",  # denoiser
            "input_dim": 3,
            "hidden_dim": 32,
            "output_dim": 2,
            "hidden_layers": 4,

            # output_params
            "log_dir": "./logs/",

            # noise_scheduler
            "beta_start": 1e-5,
            "beta_end": 1e-2,
            "timesteps": 50,
        })


    # def get_config(self):
    #     return self.__dict__.update(self.cfg)

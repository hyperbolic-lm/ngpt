class Validator:
    @staticmethod
    def validate(config):
        assert config['use_nGPT'] in (0, 1), "use_nGPT must be 0 (GPT) or 1 (nGPT)."
        if config['use_nGPT'] == 0:
            assert config['weight_decay'] == 0.1, "weight_decay = 0.1, if use_nGPT is False."
            assert config['warmup_iters'] == 2000, "warmup_iters = 2000, if use_nGPT is False."
        else:
            assert config['weight_decay'] == 0.0, "weight_decay = 0.0, if use_nGPT is True."
            assert config['warmup_iters'] == 0, "warmup_iters = 0, if use_nGPT is True."
        return config

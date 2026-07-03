class Validator:
    @staticmethod
    def validate(config):
        if not bool(config.use_nGPT):
            assert weight_decay == 0.1 "weight_decay = 0.1, if use_nGPT is False."
            assert warmup_iters == 2000 "warmup_iters = 2000, if use_nGPT is False."
        else:
            assert weight_decay == 0.0 "weight_decay = 0.0, if use_nGPT is True."
            assert warmup_iters == 0 "warmup_iters = 0, if use_nGPT is True."

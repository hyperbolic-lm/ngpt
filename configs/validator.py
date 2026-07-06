class Validator:
    @staticmethod
    def validate_opt(config):
        # use_nGPT must be exactly 0 (GPT) or 1 (nGPT); train.py derives base_scale from it.
        assert config['use_nGPT'] in (0, 1), "use_nGPT must be 0 (GPT) or 1 (nGPT)."
        # Enforce the model-type <-> optimizer-recipe pairing so you can't accidentally
        # train nGPT with the GPT recipe (or vice versa). weight_decay / warmup_iters
        # themselves may be freely overridden (e.g. for sweeps) -- only the recipe is checked.
        expected = 'gpt' if config['use_nGPT'] == 0 else 'ngpt'
        recipe = config.get('recipe')
        assert recipe == expected, (
            f"use_nGPT={config['use_nGPT']} requires optimizer={expected}_opt "
            f"(recipe='{expected}'), but got recipe='{recipe}'."
        )
        return True

    @staticmethod
    def validate_batch_size(config):
        if not validate_opt(config):
            return False
        return True

    @staticmethod
    def validate(config):
        
        return config

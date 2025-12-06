from lm_polygraph.stat_calculators.hidden_stat_calculator import HiddenStateCalculator


def load_stat_calculator(config, builder):
        return HiddenStateCalculator(
        output_attentions=config.output_attentions,
        output_hidden_states=config.output_hidden_states,
        n_alternatives=10,
        target_layer=config.target_layer,
    )

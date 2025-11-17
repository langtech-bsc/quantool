from huggingface_hub import ModelCard, ModelCardData

from quantool.core.meta import TemplateQuantizationCard


def create_model_card_from_template(template: TemplateQuantizationCard) -> ModelCard:
    """Create a Hugging Face ModelCard from a TemplateQuantizationCard."""
    # build the ModelCardData payload
    card_data = ModelCardData(
        # use title as the model name
        name=template.title,
        tags=["quantization"],
        description=template.description,
        metrics=template.hyperparameters,
        intended_use=template.intended_use,
        limitations=template.limitations,
        citations=template.citations,
    )
    # pass the card_data into from_template
    return ModelCard.from_template(card_data=card_data, template_name=template.title)

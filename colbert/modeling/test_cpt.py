from modeling_cpt import CPTForConditionalGeneration, CPTForSequenceClassification
from transformers import BartForConditionalGeneration, BartForSequenceClassification
from transformers import BertTokenizer

model_dict = {
    'cpt': (CPTForConditionalGeneration, "fnlp/cpt-base", "../../../../../pretrain/cpt/"),
    "cptbart": (BartForConditionalGeneration, "fnlp/bart-base-chinese", "../../../../../pretrain/cptbart/")
}


def load_model(model_type='cptbart'):
    print('loading model')
    model_class, pretrain_model, path = model_dict[model_type]
    tokenizer = BertTokenizer.from_pretrained(pretrain_model, cache_dir=path)
    model = model_class.from_pretrained(pretrain_model, cache_dir=path)
    print('model loaded')
    return tokenizer, model


def test_generation():
    tokenizer, model = load_model()
    input_ids = tokenizer.encode("", return_tensors='pt')
    # input(inputs)
    # input_ids = inputs["input_ids"]
    pred_ids = model.generate(input_ids, num_beams=4, max_length=20)
    print(tokenizer.convert_ids_to_tokens(pred_ids[0]))
    # ['[SEP]', '[CLS]', '北', '京', '是', '中', '国', '的', '首', '都', '[SEP]']


if __name__ == '__main__':
    test_generation()

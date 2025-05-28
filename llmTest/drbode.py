import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Nosso modelo base
model_name = "recogna-nlp/internlm-chatbode-7b" 

# Configuração para quantização do nosso modelo
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype = compute_dtype,
    bnb_4bit_use_double_quant = False,
)


# Carregando modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    trust_remote_code = True,
    device_map = 'auto'     
)

original_model = original_model.eval()


# Perguntas mandadas para o Dr Bode
evaluation_inputs = [
    'Apareceram aftas na minha boca e uma amiga disse que posso usar nistatina oral para tratar. Para que serve e como usar nistatina oral? É indicado para tratar aftas?',
    'Estou com dor no corpo, dor de cabeça, febre alta e um forte cansaço. O que pode ser? Devo tomar algum remédio?',
    'Me explique, detalhadamente, qual a diferença entre uma gripe e um resfriado.'
]

## Carregando o Dr Bode
from peft import PeftModel, PeftConfig
model = PeftModel.from_pretrained(original_model, 'recogna-nlp/doutor-bode-7b-360k')
model = model.eval()

## Realizando a inferência e verificando as respostas
for q in evaluation_inputs:
    print(q)
    response, _ = model.chat(tokenizer, q, do_sample=False, history=[])
    print(response)
    print()

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig

llm_model = 'recogna-nlp/bode-7b-alpaca-pt-br'
config = PeftConfig.from_pretrained(llm_model)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, return_dict=True, load_in_8bit=True, device_map='auto', token=hf_auth)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, token=hf_auth)
model = PeftModel.from_pretrained(model, llm_model) # Caso ocorra o seguinte erro: "ValueError: We need an `offload_dir`... Você deve acrescentar o parâmetro: offload_folder="./offload_dir".
model.eval()

def generate_prompt(instruction, input=None):
    if input:
        return f"""Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{instruction}

### Entrada:
{input}

### Resposta:"""
    else:
        return f"""Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{instruction}

### Resposta:"""
     

generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=2,
    do_sample=True
)

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=590
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Resposta:", output.split("### Resposta:")[1].strip())

inst = '''Utilize o seguinte dataset para responder os questionamentos:
IDADE, TIPO_TRABALHO, REPRESENTATIVIDADE, NIVEL_ESCOLARIDADE, ANOS_CURSADOS, ESTADO_CIVIL, PROFISSAO, RELACIONAMENTO, RACA, GENERO, GANHO_CAPITAL, PERCA_CAPITAL, HORAS_TRABALHADAS_SEMANA, PAIS_ORIGEM, RECEITA
50,Autonômo,83311,Ensino Superior Completo,13,Casado,Líder Executivo ou Gerente Executivo,Marido,Branco,Homem,0,0,13,Estados Unidos,<=50K
38,Setor Privado,215646,Ensino Médio Completo,9,Divorciado,Serviços Gerais,Não pertence à família,Branco,Homem,0,0,40,Estados Unidos,<=50K53,Setor Privado,234721,Ensino Médio Incompleto,7,Casado,Serviços Gerais,Marido,Negro,Homem,0,0,40,Estados Unidos,<=50K
28,Setor Privado,338409,Ensino Superior Completo,13,Casado,Profissional Especializado,Esposa,Negro,Mulher,0,0,40,Cuba,<=50K

Importante explicar o processo de pensamento para responder
'''

prompt = '''
Defina se essa persona:
37,Setor Privado,284582,Pós-graduação - Mestrado,14,Casado,Líder Executivo ou Gerente Executivo,Esposa,Branco,Mulher,0,0,40,Estados Unidos

Deve ou não ganhar mais que >50k.
'''

evaluate(inst,prompt)
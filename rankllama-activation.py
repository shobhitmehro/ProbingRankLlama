import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
from huggingface_hub import login
from sequences import load_ms_marco_data, load_MIND_data, IDContext, load_relevance_data
from tqdm import tqdm




n_layers = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer_name =  'meta-llama/Llama-2-7b' 
model_name = "castorini/rankllama-v1-7b-lora-passage" 
output_dir = "relevant/rankllama7b/activations" 
nqueries = 100
ndocs = 91
token_id = "use appropriate token id"

activations = {}
handle = []

def quantize_neurons(activation_tensor, output_precision=4):
    activation_tensor = activation_tensor.to(torch.float32)
    min_vals = activation_tensor.min(dim=0)[0]
    max_vals = activation_tensor.max(dim=0)[0]
    num_quant_levels = 2**output_precision
    scale = (max_vals - min_vals) / (num_quant_levels - 1)
    zero_point = torch.round(-min_vals / scale)
    quant =  torch.quantize_per_channel(activation_tensor, scale, zero_point, 1, torch.qint8)
    dequantized = quant.dequantize()
    aggregated_quant = dequantized.mean(dim=0) #mean aggregation over tokens within a sequence (can do max)
    #print(aggregated_quant.shape)
    return aggregated_quant

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model

def get_activation(name):
    def hook(model, input, output):
        # print('LAYER_OUTPUT_SHAPE',output.shape)
        qid, doc_id = IDContext.get()
        activations[name] = output[0].detach().cpu().to(torch.float16)
        output_path = f'{output_dir}/q{qid}/d{doc_id}{name}.pt'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(quantize_neurons(activations[name]), output_path)
    return hook

login(token_id)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = get_model(model_name)
print(model)

for i in range(n_layers):
    
    if 'llama' not in model_name :
        mlp_layer = model.gpt_neox.layers[i].mlp  
    else :
        mlp_layer = model.model.layers[i].mlp
        
    handle.append(mlp_layer.register_forward_hook(get_activation(f'layer_{i}_activations')))

# query_dict = load_ms_marco_data(n_queries = nqueries,n_docs=ndocs)
query_dict = load_MIND_data(51,91)

print(f"Done loading relevance data total queries: {len(query_dict)}")

for i, (query, docs) in enumerate(tqdm(query_dict.items())):
    for j, doc in enumerate(docs):
        IDContext.set(i, j)
        inputs = tokenizer(f'query: {query}', f'document: {doc}', return_tensors='pt').to(device)
        outputs = model(**inputs)

for h in handle:
    h.remove()
 


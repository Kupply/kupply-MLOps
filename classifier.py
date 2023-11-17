import torch
import torch.nn.functional as F
from torch.nn.functional import softmax

import transformers
from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer # 별도의 라이브러리 처리 필요
import sentencepiece

# from transformers import BertTokenizerFast
# from transformers import BertForSequenceClassification, AlbertForSequenceClassification

def classifier(config, dataLoader) -> int:
    
    # Load config_info
    model_path = config.model_path
    
    # Declare model and load pre-trained weights
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1',num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()  
    
    # Get the single batch from the DataLoader
    batch = next(iter(dataLoader))
    contents = batch['content']
    attention_masks = batch['attention_mask']

    with torch.no_grad():
        
        # Perform inference
        outputs = model(contents, attention_mask=attention_masks)

        # Process the outputs (e.g., applying softmax and getting predictions)
        probabilities = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    return prediction.item()  # Return the single prediction
    
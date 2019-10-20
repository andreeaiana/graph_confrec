# -*- coding: utf-8 -*-
import os
import torch
import logging
logging.basicConfig(level=logging.INFO)


class EmbeddingsParser():

    path_vocabulary = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "..", "data", "external",
                                   "scibert_scivocab_uncased", "vocab.txt")
    path_model = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "..", "..", "data", "external",
                              "scibert_scivocab_uncased", "pytorch_model.bin")
    path_configuration = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "data", "external",
            "scibert_scivocab_uncased", "bert_config.json")

    def __init__(self, gpu):
        if torch.cuda.is_available() and gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print("Using GPU device: {}.".format(str(gpu)))
        from transformers import BertConfig, BertModel, BertTokenizer
        print("Initializing pretrained SciBERT model.")
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(self.path_vocabulary)

        # Load pre-trained model (weights)
        configuration = BertConfig.from_json_file(self.path_configuration)
        configuration.output_hidden_states = True
        self.model = BertModel.from_pretrained(self.path_model,
                                               config=configuration)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        print("SciBERT model initialized.")

        self.embedding_types = {
                "AVG_L": "_average_tokens_last_layer",
                "AVG_2L": "_average_tokens_second_to_last_layer",
                "AVG_SUM_L4": "_average_tokens_sum_last_four_layers",
                "AVG_SUM_ALL": "_average_tokens_sum_all_layers",
                "AVG_CONC_L4": "_average_concat_last_four_layers",
                "MAX_2L": "_max_tokens_second_to_last_layer",
                "MAX_CONC_L4": "_max_tokens_sum_last_four_layers",
                "CONC_AVG_MAX_2L": "_concat_avg_max_tokens_second_to_last_layer",
                "CONC_AVG_MAX_SUM_L4": "_concat_avg_max_sum_last_four_layers",
                "SUM_L": "_sum_last_layer",
                "SUM_2L": "_sum_second_to_last",
                "SUM_CONC_L4": "_sum_concat_last_four_layers"
                }

    def embed_sequence(self, sequence, embedding_type):
        # Tokenize input sequence, add special tokens, covert tokens to ids
        tokenized_sequence = self.tokenizer.encode(sequence,
                                                   add_special_tokens=True)
        if len(tokenized_sequence) > 512:
            tokenized_sequence = tokenized_sequence[:512]
        # Convert tokenized sequence to PyTorch tensors
        tokens_tensor = torch.tensor([tokenized_sequence])

        # Put the model on GPU, if available
        if torch.cuda.is_available():
            tokens_tensor = tokens_tensor.to("cuda")
            self.model.to("cuda")

        # predict the hidden states features for each layer
        with torch.no_grad():
            outputs = self.model(tokens_tensor)[-1]
            encoded_layers = outputs[1:]

        # Check output size (batch size, sequence lenght, model hidden
        # dimension)
        try:
            assert tuple(
                    encoded_layers[0].shape) == (1, len(tokenized_sequence),
                                                 self.model.config.hidden_size)
        except Exception as e:
            print("Wrong output size: {}.".format(e))

        # Convert the hidden state embeddings into single token vectors
        # Holds the list of 12 layers embeddings forn each token
        # Size: [#tokens, #layers, #features]
        self.token_embeddings = []
        for token_idx in range(len(tokenized_sequence)):
            hidden_layers = []
            for layer_idx in range(len(encoded_layers)):
                vector = encoded_layers[layer_idx][0][token_idx]
                hidden_layers.append(vector)
            self.token_embeddings.append(hidden_layers)
        self.token_embeddings = self.token_embeddings[1:-1]
        embedding_function = self.__getattribute__(
                self.embedding_types[embedding_type])
        sequence_embedding = embedding_function()
        return sequence_embedding.cpu().numpy()

    def _last_layer(self):
        return [torch.stack(layer)[-1] for layer in self.token_embeddings]

    def _second_to_last_layer(self):
        return [torch.stack(layer)[-2] for layer in self.token_embeddings]

    def _sum_last_four_layers(self):
        return [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                self.token_embeddings]

    def _sum_all_layers(self):
        return [torch.sum(torch.stack(layer)[:], 0) for layer in
                self.token_embeddings]

    def _concat_last_four_layers(self):
        return [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0)
                for layer in self.token_embeddings]

    def _average_tokens_last_layer(self):
        return torch.mean(torch.stack(self._last_layer()), 0)

    def _average_tokens_second_to_last_layer(self):
        return torch.mean(torch.stack(self._second_to_last_layer()), 0)

    def _average_tokens_sum_last_four_layers(self):
        return torch.mean(torch.stack(self._sum_last_four_layers()), 0)

    def _average_tokens_sum_all_layers(self):
        return torch.mean(torch.stack(self._sum_all_layers()), 0)

    def _average_concat_last_four_layers(self):
        return torch.mean(torch.stack(self._concat_last_four_layers()), 0)

    def _max_tokens_second_to_last_layer(self):
        return torch.max(torch.stack(self._second_to_last_layer()), 0).values

    def _max_tokens_sum_last_four_layers(self):
        return torch.max(torch.stack(self._sum_last_four_layers()), 0).values

    def _concat_avg_max_tokens_second_to_last_layer(self):
        return torch.cat((self._average_tokens_second_to_last_layer(),
                          self._max_tokens_second_to_last_layer()))

    def _concat_avg_max_sum_last_four_layers(self):
        return torch.cat((self._average_tokens_sum_last_four_layers(),
                          self._max_tokens_sum_last_four_layers()))

    def _sum_last_layer(self):
        return torch.sum(torch.stack(self._last_layer()), 0)

    def _sum_second_to_last(self):
        return torch.sum(torch.stack(self._second_to_last_layer()), 0)

    def _sum_concat_last_four_layers(self):
        return torch.sum(torch.stack(self._concat_last_four_layers()), 0)

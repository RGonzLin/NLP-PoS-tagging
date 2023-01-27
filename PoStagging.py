# Import libraries
import conllu
import numpy as np
import tensorflow as tf


class PoStaggingModel():

    def GetSentences(self,path):

        # Open the plain text file for reading; assign under 'data'
        with open(path, mode="r", encoding="utf-8") as data:

            # Read the file contents and assign under 'annotations'
            annotations = data.read()

        # Use the parse() function to parse the annotations; store under 'sentences'
        sentences = conllu.parse(annotations)

        # Return sentences
        return sentences

    def InputEncoding(self,sentences,trainSet=False,predictSet=False): 

        # Initialize array to fill with input values
        input = []

        # Fill array
        for sentence in sentences:
            currentInput = []
            for i in range(len(sentence[:])):
                currentInput.append(sentence[i]['form'])
            input.append(currentInput)

        # Let only the train set tokenizer be available to other methods
        if trainSet==True:

            # Create tokenizer object
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='OOV') 

            # Build vocabularies for set
            self.tokenizer.fit_on_texts(input)

        # Encode inputs for set
        inputEncoded = self.tokenizer.texts_to_sequences(input) 

        # Let only the pre-padded input be availible to other methods when encodig a set to be used for prediction
        # This is used in the predict method to prune the padding post-prediction
        if predictSet==True:
            self.inputEncoded = inputEncoded

        # Pad inputs for set
        inputEncodedPadded = tf.keras.utils.pad_sequences(inputEncoded, maxlen=128, padding='post',
                                                                truncating='post')

        # Return the encoded and padded set
        return inputEncodedPadded

    def TargetEncoding(self,sentences):

        # Initialize array to fill with input and output values
        output = []

        # Fill array
        for sentence in sentences:
            currentOutput = []
            for i in range(len(sentence[:])):
                currentOutput.append(sentence[i]['upos'])
            output.append(currentOutput)

        # Create dictionary to assign a number to each unique UPOS (both 'X' and '_' are considered to correspond to
        # 'other' and therefore are assigned the same number)
        assignedUPOS = {'ADJ':1 ,'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 
                        'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, 
                        '_': 17}

        # Initialize array to fill with encoded outputs
        outputEncoded = []

        # Encode
        for i in range(len(output)):
            currentOutput = []
            for upos in output[i]:
                currentOutput.append(assignedUPOS[upos])
            outputEncoded.append(currentOutput)

        # Pad outputs for set
        outputEncodedPadded = tf.keras.utils.pad_sequences(outputEncoded, maxlen=128,padding='post',
                                                            truncating='post')

        #return outputEncodedPaddedCategorical
        return outputEncodedPadded

    def CreateModel(self,embeddingSize=64,lstmUnits=32,dropout=0.0,printSummary=False):

        # Get size of vocabulary
        vsize = len(self.tokenizer.word_index)+1

        # Define inputs
        inputs = tf.keras.Input(shape=(128,), dtype=tf.int32) # Each sentence has 128 elements, and batch size is left
                                                            # empty as it should work for any size

        # Create network 
        x = tf.keras.layers.Embedding(vsize,embeddingSize,mask_zero=True)(inputs) # Default: 64 numbers per word
            # Create LSTM layer
        lstm = tf.keras.layers.LSTM(lstmUnits,return_sequences=True,dropout=dropout) # Default: 32 LSTM units
        # Continue with network creation
        x = tf.keras.layers.Bidirectional(lstm)(x)
            # Create desnse layer
        dense = tf.keras.layers.Dense(17,activation='softmax') # 17 labels 
        # Continue with network creation
        outputs = tf.keras.layers.TimeDistributed(dense)(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs,outputs=outputs)

        # Print model summary if requested
        if printSummary==True:
            print(self.model.summary())

        # Return the created model
        return self.model

    def Train(self,trainInputs,trainTargets,inputValidation=None,targetValidation=None,epochs=10,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),metrics=['accuracy'],batchSize=None):

        # Compile model
        self.model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=metrics) 

        # If both input and target sets for validation were provided
        if type(inputValidation) is np.ndarray and type(targetValidation) is np.ndarray:
            
            # Train model
            self.trainedModel = self.model.fit(trainInputs,trainTargets,epochs=epochs,validation_data=(inputValidation,targetValidation),
                                                batch_size=batchSize)
        
        # If input or target sets for validation were not provided
        else:

            # Train model
            self.trainedModel = self.model.fit(trainInputs,trainTargets,epochs=epochs,batch_size=batchSize)
        
        # Return trained model
        return self.trainedModel

    def Test(self,inputTest,targetTest):

        # Test model 
        testedModel = self.model.evaluate(inputTest,targetTest)

        # Return tested model
        return testedModel

    def Predict(self,inputSample,prunePadding=False):

        # Predict UPOS given a text sample
        predictionsRaw = self.model.predict(inputSample)

        # Initialize array to contain dictionary keys
        predictionsKeys = []

        # Fill array
        for sentence in predictionsRaw:
            currentSentence = []
            for word in sentence:
                currentSentence.append(np.argmax(word))
            predictionsKeys.append(currentSentence)

        # Decoding dictionary 
        assignedUPOS = {1: 'ADJ', 2: 'ADP', 3: 'ADV', 4: 'AUX', 5: 'CCONJ', 6: 'DET', 7: 'INTJ', 8: 'NOUN', 9: 'NUM', 
                10: 'PART', 11: 'PRON', 12: 'PROPN', 13: 'PUNCT', 14: 'SCONJ', 15: 'SYM', 16: 'VERB', 17: 'X'} 

       # Initialize array to contain predicted UPOS
        predictionDecoded = []
        
        # Decode
        for i in range(len(predictionsKeys)):
            currentSentence = []
            for upos in predictionsKeys[i]:
                currentSentence.append(assignedUPOS[upos])
            predictionDecoded.append(currentSentence)

        # If pruning the predictions is desired (only when InputEncoding method was used)
        if prunePadding==True:
            predictionsPruned = []
            for i in range(len(predictionDecoded)):
                predictionsPruned.append(predictionDecoded[i][:len(self.inputEncoded[i])])
            predictionDecoded = predictionsPruned

        # Return decoded 
        return predictionDecoded

    def Save(self,path):
        
        # Save model in the specified path
        return self.model.save(path)

    def Load(self,path):

        # Load model in the specified path
        self.model = tf.keras.models.load_model(path)

        # Return model
        return self.model
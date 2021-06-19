import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_dim, gru_dim, dropout):
        super(EncoderLayer, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.gru = tf.keras.layers.GRU(gru_dim, dropout=dropout, return_state=True)

    def call(self, x, mask, training):

        embeddings = self.embedding(x)
        outputs, state = self.gru(embeddings,
                                  mask=mask,
                                  training=training)

        return outputs, state


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_layer, gru_dim, start_token, end_token,
                 dropout=0.1, maximum_iterations=50):
        super(DecoderLayer, self).__init__()

        self.start_token, self.end_token = start_token, end_token
        self.embedding = emb_layer
        self.gru_cell = tf.keras.layers.GRUCell(gru_dim, dropout=dropout)
        self.output_layer=tf.keras.layers.Dense(vocab_size)
        self.maximum_iterations = maximum_iterations
        self.decoder = tfa.seq2seq.BasicDecoder(self.gru_cell,
                                                tfa.seq2seq.TrainingSampler(),
                                                self.output_layer)
    

    @tf.function
    def call(self, encoder_outputs, encoder_state, y, mask, training):
        predictions_arr = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # , clear_after_read=False
        if training is True:
            time_len = tf.shape(y)[1]
            embeddings = self.embedding(y)
            mask = tf.cast(mask, tf.int32)
            ta = tf.TensorArray(tf.float32, size=time_len, dynamic_size=False)
            for i in range(time_len):
                ta.write(i, encoder_outputs)
            encoder_outputs = tf.transpose(ta.stack(), [1, 0, 2])
            inputs = tf.concat([embeddings, encoder_outputs], axis=-1)
            predictions, _, _ = self.decoder(inputs,
                                             sequence_length=mask,
                                             initial_state=encoder_state,
                                             training=training)
            predictions = predictions.rnn_output
        else:
            batch_size = tf.shape(x)[0]
            outputs = tf.fill([batch_size], self.start_token)
            finished = tf.math.equal(outputs, self.end_token)
            iteration = 0
            state = encoder_state
            while not tf.reduce_all(finished) and iteration < self.maximum_iterations:
                outputs = self.embedding(outputs)
                outputs = tf.concat([outputs, encoder_outputs], axis=-1)
                outputs, state = self.gru_cell(outputs, state, training=training)
                outputs = self.output_layer(outputs)
                outputs = tf.math.argmax(tf.nn.softmax(outputs), axis=-1, output_type=tf.int32)
                predictions_arr = predictions_arr.write(iteration, outputs)
                iteration += 1
                finished = tf.math.equal(outputs, self.end_token)
            predictions = tf.transpose(predictions_arr.stack(), [1, 0])

        return predictions


class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, encoder_dim, start_token,
                 end_token, dropout=0.1, maximum_iterations=50):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = EncoderLayer(vocab_size, emb_dim, encoder_dim, dropout)
        decoder_dim = encoder_dim
        self.decoder = DecoderLayer(vocab_size, self.encoder.embedding, decoder_dim,
                                    start_token, end_token, dropout, maximum_iterations)


    def call(self, x, y, inputs_mask, outputs_mask, training):
        encoder_outputs, state = self.encoder(x, inputs_mask, training)
        outputs = self.decoder(encoder_outputs, state, y, outputs_mask, training)
        return outputs


@tf.function(experimental_relax_shapes=True)
def train_step(X, Y_input, Y_output, X_lens, Y_lens):
#     X, Y_input, Y_output = tf.convert_to_tensor(X), tf.convert_to_tensor(Y_input), tf.convert_to_tensor(Y_output)
    with tf.GradientTape() as tape:
        predictions = model(X, Y_input, tf.sequence_mask(X_lens),
                            Y_lens, training=True)
        loss = loss_fn(Y_output, predictions, tf.sequence_mask(Y_lens))
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


model = Seq2SeqModel(len(word2id), 20, 256,
                     start_token=word2id['^'],
                     end_token=word2id['$'])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

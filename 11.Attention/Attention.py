# [MXDL-11-04] Attention.py
# Attention Networks for time series prediction
# [1] Minh-Thang Luong et, al., 2015, Effective Approaches 
# to Attention-based Neural Machine Translation
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/eyXHpL4dlYU
#
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Dot, Activation
from tensorflow.keras.layers import Concatenate, Reshape

class AttentionLayer:
    def __init__(self, n_hidden):
        self.attentionFFN = Dense(n_hidden, activation='tanh')
        
    def __call__(self, d, e):
        dot_product = Dot(axes=(2, 2))([d, e])
        score = Activation('softmax')(dot_product)
        value = Dot(axes=(2, 1))([score, e])
        concat = Concatenate()([value, d])
        h_attention = self.attentionFFN(concat)
        return h_attention    # attentional hidden state
       
class Encoder:
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        self.encoderGRU = GRU(n_hidden,
                              return_sequences=True,
                              return_state = True)
    
    def __call__(self, x):
        return self.encoderGRU(x)

# [1] 3.3 Input-feeding approach
# Attentional vectors are concatenated with inputs at the next time steps.
class Decoder:
    def __init__(self, n_hidden, n_feed):
        self.n_hidden = n_hidden
        self.n_feed = n_feed
        self.decoderGRU = GRU(n_hidden)
        self.inputFeedingFFN = Dense(n_feed, activation='tanh')
        self.attention = AttentionLayer(n_hidden)
    
    def __call__(self, x, o_enc, h_enc):
        outputs = []   # outputs of decoder (many-to-many)
        i_feed = tf.zeros(shape=(tf.shape(x)[0], self.n_hidden))
        for t in range(x.shape[1]):
            i_cat = self.inputFeedingFFN(i_feed)
            i_cat = Concatenate()([i_cat, x[:, t, :]])
            i_cat = Reshape([1, -1])(i_cat)
            h_dec = self.decoderGRU(i_cat, initial_state = h_enc)
            
            # Find attentional hidden state
            h_att = self.attention(Reshape((1, -1))(h_dec), o_enc)
            
            # Update encoder's hidden state and the input-feeding 
            # vector for the next step
            h_enc = h_dec
            i_feed = Reshape((-1,))(h_att)
            
            # Collect outputs at all time steps.
            outputs.append(Reshape((self.n_hidden,))(h_att))
            
        outputs = tf.convert_to_tensor(outputs)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        
        return outputs
    
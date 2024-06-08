import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn

# Keras 모델 정의
class UserMovieEmbeddingKeras(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbeddingKeras, self).__init__()
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        uemb = self.u_embedding(x[:, 0])
        memb = self.m_embedding(x[:, 1])
        m_u = self.m_u_merge([uemb, memb])
        return self.m_u_fc(m_u)

# PyTorch 모델 정의
class UserMovieEmbedding(nn.Module):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        self.u_embedding = nn.Embedding(num_embeddings=len_users, embedding_dim=embedding_dim)
        self.m_embedding = nn.Embedding(num_embeddings=len_movies, embedding_dim=embedding_dim)
        self.m_u_fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        user_ids, movie_ids = x[:, 0], x[:, 1]
        uemb = self.u_embedding(user_ids)
        memb = self.m_embedding(movie_ids)
        m_u = (uemb * memb).sum(dim=1, keepdim=True)
        return torch.sigmoid(self.m_u_fc(m_u))

if __name__ == "__main__":
    users_num, items_num, embedding_dim = 6041, 3953, 100

    # Keras 모델 불러오기
    embedding_save_file_dir = 'save_weights/user_movie_embedding_case4.h5'
    keras_model = UserMovieEmbeddingKeras(users_num, items_num, embedding_dim)
    
    # 임의의 입력 데이터로 모델 호출하여 변수들 생성
    dummy_input = tf.constant([[0, 0]], dtype=tf.int32)
    _ = keras_model(dummy_input)

    # Keras 모델 가중치 불러오기
    keras_model.load_weights(embedding_save_file_dir)

    # PyTorch 모델 초기화
    pytorch_model = UserMovieEmbedding(users_num, items_num, embedding_dim)

    # Keras 모델의 가중치 가져오기
    keras_weights = keras_model.get_weights()

    # Keras 가중치를 PyTorch 모델에 적용
    with torch.no_grad():
        pytorch_model.u_embedding.weight.copy_(torch.tensor(keras_weights[0]))
        pytorch_model.m_embedding.weight.copy_(torch.tensor(keras_weights[1]))
        pytorch_model.m_u_fc.weight.copy_(torch.tensor(keras_weights[2].T))
        pytorch_model.m_u_fc.bias.copy_(torch.tensor(keras_weights[3]))

    # PyTorch 모델 저장
    torch.save(pytorch_model.state_dict(), 'user_movie_embedding_case4.pth')

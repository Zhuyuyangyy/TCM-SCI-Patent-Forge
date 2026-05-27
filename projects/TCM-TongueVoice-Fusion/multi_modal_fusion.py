"""
多模态融合模块 - M01 舌象-声纹-面色三模态体质识别
支持各模态独立子网络 + 融合预测
"""
import numpy as np


class MultiModalFusion:
    """
    三模态体质识别融合网络
    舌象(8d) + 声纹(8d) + 面色(6d) -> 22d融合 -> 9类体质
    """
    CONSTITUTIONS = ['气虚', '阳虚', '阴虚', '痰湿', '湿热', '血瘀', '气郁', '特禀', '平和']

    def __init__(self, tongue_dim=8, voice_dim=8, face_dim=6,
                 hidden_dims=[64, 32], output_dim=9):
        self.tongue_dim = tongue_dim
        self.voice_dim = voice_dim
        self.face_dim = face_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.total_dim = tongue_dim + voice_dim + face_dim  # 22

        # 子网络：独立处理各模态
        self.sub_weights = {}
        self.sub_biases = {}
        self._init_sub_network('tongue', tongue_dim, 32)
        self._init_sub_network('voice', voice_dim, 32)
        self._init_sub_network('face', face_dim, 16)

        # 融合网络
        self.fusion_weights = []
        self.fusion_biases = []
        prev_dim = 32 * 3 + 16  # 112 = sub-net outputs concatenated
        for h_dim in hidden_dims:
            scale = np.sqrt(2.0 / (prev_dim + h_dim))
            self.fusion_weights.append(
                np.random.randn(prev_dim, h_dim) * scale)
            self.fusion_biases.append(np.zeros(h_dim))
            prev_dim = h_dim
        # 输出层
        scale = np.sqrt(2.0 / (prev_dim + output_dim))
        self.fusion_weights.append(
            np.random.randn(prev_dim, output_dim) * scale)
        self.fusion_biases.append(np.zeros(output_dim))

        self.training_history = []
        self._init_modality_output()

    def _init_sub_network(self, name, input_dim, output_dim):
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.sub_weights[name] = np.random.randn(input_dim, output_dim) * scale
        self.sub_biases[name] = np.zeros(output_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-10)

    def _forward_sub(self, X, name):
        """单个模态子网络前向"""
        z = np.dot(X, self.sub_weights[name]) + self.sub_biases[name]
        return self._relu(z)

    def _forward_fusion(self, X):
        """融合网络前向"""
        current = X
        for i in range(len(self.fusion_weights) - 1):
            z = np.dot(current, self.fusion_weights[i]) + self.fusion_biases[i]
            current = self._relu(z)
        z_out = np.dot(current, self.fusion_weights[-1]) + self.fusion_biases[-1]
        return self._softmax(z_out)

    def forward(self, X):
        """完整22-dim输入的前向传播（兼容旧API）"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self._forward_fusion(X)

    def predict(self, X):
        """旧API：22-dim完整特征输入"""
        return self.forward(X)

    def predict_modality(self, features, modality):
        """
        单模态预测 - 使用该模态的子网络
        modality: 'tongue' | 'voice' | 'face'
        """
        feat = np.array(features).flatten().reshape(1, -1)
        sub_out = self._forward_sub(feat, modality)
        # 融合网络输出
        return self._forward_fusion(sub_out)[0]

    def predict_fusion(self, tongue_feat, voice_feat, face_feat):
        """
        三模态融合预测
        """
        t = np.array(tongue_feat).flatten().reshape(1, -1)
        v = np.array(voice_feat).flatten().reshape(1, -1)
        f = np.array(face_feat).flatten().reshape(1, -1)

        t_out = self._forward_sub(t, 'tongue')
        v_out = self._forward_sub(v, 'voice')
        f_out = self._forward_sub(f, 'face')

        concat = np.concatenate([t_out, v_out, f_out], axis=1)
        return self._forward_fusion(concat)[0]

    def predict_class(self, X):
        probs = self.predict(X)
        idx = np.argmax(probs)
        return self.CONSTITUTIONS[idx], probs[idx]

    def train_step(self, tongue_data, voice_data, face_data, labels,
                   learning_rate=0.01):
        """
        单步训练（需外部数据）
        labels: one-hot (n_samples, 9)
        """
        n = len(labels)
        outputs = []
        for i in range(n):
            t = np.array(tongue_data[i]).flatten().reshape(1, -1)
            v = np.array(voice_data[i]).flatten().reshape(1, -1)
            f = np.array(face_data[i]).flatten().reshape(1, -1)
            t_out = self._forward_sub(t, 'tongue')
            v_out = self._forward_sub(v, 'voice')
            f_out = self._forward_sub(f, 'face')
            concat = np.concatenate([t_out, v_out, f_out], axis=1)
            out = self._forward_fusion(concat)
            outputs.append(out)
        Y_pred = np.vstack(outputs)

        # Cross-entropy loss gradient
        eps = 1e-10
        delta = (Y_pred - labels) / n

        # Backprop through fusion layers
        grad = delta
        activations = []
        t_acts, v_acts, f_acts = [], [], []
        for i in range(n):
            t = np.array(tongue_data[i]).flatten().reshape(1, -1)
            v = np.array(voice_data[i]).flatten().reshape(1, -1)
            f = np.array(face_data[i]).flatten().reshape(1, -1)
            t_act = self._forward_sub(t, 'tongue')
            v_act = self._forward_sub(v, 'voice')
            f_act = self._forward_sub(f, 'face')
            t_acts.append(t_act)
            v_acts.append(v_act)
            f_acts.append(f_act)
            activations.append(np.concatenate([t_act, v_act, f_act], axis=1))

        acts = np.vstack(activations)
        # Output layer
        delta_out = delta
        dW_last = np.dot(acts.T, delta_out)
        db_last = np.dot(np.ones(n), delta_out)
        grad = np.dot(delta_out, self.fusion_weights[-1].T)

        # Hidden layers (reverse)
        for i in range(len(self.fusion_weights) - 2, -1, -1):
            z = np.dot(acts, self.fusion_weights[i]) + self.fusion_biases[i]
            relu_grad = (z > 0).astype(float)
            delta_h = grad * relu_grad
            dW = np.dot(acts.T, delta_h)
            db = np.dot(np.ones(n), delta_h)
            self.fusion_weights[i] -= learning_rate * dW
            self.fusion_biases[i] -= learning_rate * db
            if i > 0:
                grad = np.dot(delta_h, self.fusion_weights[i].T)

        self.fusion_weights[-1] -= learning_rate * dW_last
        self.fusion_biases[-1] -= learning_rate * db_last

        # Sub-network gradients
        for name, acts_list in [('tongue', t_acts), ('voice', v_acts), ('face', f_acts)]:
            sub_dim = {'tongue': self.tongue_dim, 'voice': self.voice_dim,
                       'face': self.face_dim}[name]
            sub_grad = np.dot(delta_out, self.fusion_weights[0].T)[:, :32]
            sub_grad = sub_grad * (np.vstack(acts_list) > 0).astype(float)
            grad_t = np.dot(delta_out, self.fusion_weights[0].T)
            # distribute to sub-nets
            for i in range(n):
                t_in = np.array([tongue_data[i]]).flatten() if name == 'tongue' else \
                       np.array([voice_data[i]]).flatten() if name == 'voice' else \
                       np.array([face_data[i]]).flatten()
                dW_sub = np.outer(t_in, sub_grad[i])
                self.sub_weights[name] -= learning_rate * dW_sub
                self.sub_biases[name] -= learning_rate * sub_grad[i]

        loss = -np.sum(labels * np.log(Y_pred + eps)) / n
        self.training_history.append(loss)
        return loss


def demo():
    """内置测试"""
    import numpy as np
    fusion = MultiModalFusion(
        tongue_dim=8, voice_dim=8, face_dim=6,
        hidden_dims=[64, 32], output_dim=9
    )

    # 模拟数据
    np.random.seed(42)
    tongue_data = np.random.randn(50, 8)
    voice_data = np.random.randn(50, 8)
    face_data = np.random.randn(50, 6)
    labels = np.zeros((50, 9))
    labels[np.arange(50), np.random.randint(0, 9, 50)] = 1

    # 训练几步
    for epoch in range(5):
        loss = fusion.train_step(tongue_data, voice_data, face_data, labels)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}")

    # 测试预测
    t = tongue_data[0]
    v = voice_data[0]
    f = face_data[0]

    probs = fusion.predict_fusion(t, v, f)
    print(f"\n融合预测: {fusion.CONSTITUTIONS[np.argmax(probs)]} "
          f"(置信度: {np.max(probs):.3f})")

    t_prob = fusion.predict_modality(t, 'tongue')
    print(f"舌象预测: {fusion.CONSTITUTIONS[np.argmax(t_prob)]} "
          f"(置信度: {np.max(t_prob):.3f})")


if __name__ == '__main__':
    demo()

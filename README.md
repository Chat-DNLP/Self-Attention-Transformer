# Práctica: implementación del mecanismo de auto-atención con enmascaramiento del modelo Transformer

En este repositorio se implementa de manera simple el mecanismo de auto-atención con enmascaramiento de la arquitectura Transformer. La finalidad de ellos es comprender con detalle el funcionamiento dicho mecanismo paso por paso.

## Paso 1. Cálculo de los vectores clave, consulta y valor (K,Q,V)

A partir de las matrices $W^K$, $W^Q$ y $W^V$, las cuales se ajustan durnate el entrenamiento, se calculan los vectores de clave (key), consulta (query) y valor (value). Para este caso simple, ya se disponen de los vectores de consulta, clave y valor.

```python
Q = torch.tensor([[0.0, 0.0, 0.0], [1, 1, 1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
K = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]])
V = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 1.]])
```

<div align="center">
  <img src="images/matrices.png" alt="Matrices" width =600 />
</div>

## Paso 2. Cálculo de las puntuaciones de atención

A continuación, se obtienen los *scores* de atención medinate la multiplicación de las matrices de vectores clave y consulta.

```python
scores = torch.matmul(Q, K.t())
```

<div align="center">
  <img src="images/scores.png" alt="Scores" width =600 />
</div>

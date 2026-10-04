# Detector de produtos de higiene com YOLOv7

Modelo de visão computacional que identifica **12 tipos de produto de higiene pessoal** em fotos, treinado com a rede **YOLOv7** em um dataset próprio, criado em grupo (4 pessoas). Na coleta, fiquei com os produtos de banho, proteção, álcool em gel e sabonetes, e fui responsável por treinar, avaliar e analisar o modelo.

**Resultado no conjunto de teste (65 imagens nunca vistas): mAP@.5 = 0,784 | mAP@.5:.95 = 0,568**

> Projeto acadêmico de Inteligência Artificial (ADS, UVV). Portfólio: https://isadoraeler.github.io/portfolio/

## Classes

álcool em gel, aparelho de barbear, condicionador, pasta de dente, desodorante, escova de dente, hidratante, papel higiênico, perfume, sabonete em barra, sabonete líquido e shampoo.

## Dataset

| Conjunto | Imagens |
|---|---|
| Treino | 470 |
| Validação | 136 |
| Teste | 65 |
| **Total** | **671** |

- Fotos próprias do grupo, com variação de fundo, distância e ângulo.
- Rotuladas no [MakeSense.ai](https://www.makesense.ai/), em **formato YOLO**: um `.txt` por imagem, com uma linha por objeto:

```
classe  centro_x  centro_y  largura  altura
2       0.447080  0.450957  0.535315  0.721739
```

Os valores de posição e tamanho são normalizados (0 a 1) em relação às dimensões da imagem. A classe é o índice na lista `names` do arquivo `custom.yaml`. Imagem e rótulo ficam na mesma pasta.

> O dataset e os pesos treinados não estão neste repositório (tamanho e privacidade das fotos).

## Método

- **Rede:** YOLOv7, com fine-tuning a partir dos pesos pré-treinados `yolov7.pt`.
- **Treino:** 100 épocas, batch 16, imagens 640x640, hiperparâmetros `hyp.scratch.p5.yaml`, GPU T4 no Google Colab.
- **Limiar de confiança:** 0,30, escolhido pelo pico da curva F1 (F1 = 0,52).

## Resultados no teste

| Métrica | Valor |
|---|---|
| Precisão | 0,609 |
| Recall | 0,838 |
| mAP@.5 | 0,784 |
| mAP@.5:.95 | 0,568 |

mAP@.5 por classe:

| Classe | mAP@.5 | Classe | mAP@.5 |
|---|---|---|---|
| papel higiênico | 0,923 | escova de dente | 0,805 |
| perfume | 0,927 | aparelho de barbear | 0,795 |
| pasta de dente | 0,922 | sabonete líquido | 0,757 |
| álcool em gel | 0,898 | condicionador | 0,645 |
| sabonete em barra | 0,896 | desodorante | 0,644 |
| hidratante | 0,840 | **shampoo** | **0,355** |

> O teste tem poucos exemplos por classe (4 a 14, exceto desodorante com 59), então a nota de cada classe varia bastante com um único acerto ou erro.

## Experimentos e aprendizados

| Experimento | Resultado |
|---|---|
| Treino **do zero** (sem pesos pré-treinados) | mAP@.5 = 0,16 (validação) |
| **Fine-tuning** com `yolov7.pt` (modelo final) | mAP@.5 = 0,76 (validação) / **0,784 (teste)** |
| Fine-tuning + `--image-weights` + learning rate menor | 0,764 (teste), **sem melhora** |

1. **Transfer learning foi decisivo.** Com apenas 470 imagens de treino, partir de pesos pré-treinados foi muito melhor do que treinar do zero.
2. **O ponto fraco é um trio de frascos parecidos:** shampoo, hidratante e condicionador. Pela matriz de confusão, o modelo troca um pelo outro. Essas classes têm cerca de 35 caixas cada, contra 228 do desodorante.
3. **Rebalancear não resolveu.** Mostrar mais vezes as fotos das classes raras fez o modelo decorar essas imagens (overfitting). A conclusão é que o gargalo é a **variedade das fotos**, e não quantas vezes o modelo as vê.
4. **O desodorante gera falsos positivos:** ele domina o treino e o modelo aposta nele quando fica em dúvida.

## Limitações e próximos passos

- Coletar mais fotos de shampoo, hidratante e condicionador, com ângulos, fundos e embalagens diferentes (incluindo refis em sachê).
- Testar augmentation mais forte (rotação, escala, brilho).
- Aumentar o conjunto de teste para uma avaliação por classe mais estável.

## Como reproduzir

1. Clone o [YOLOv7](https://github.com/WongKinYiu/yolov7) e baixe o `yolov7.pt`.
2. Monte um dataset no formato acima, com as pastas `treino`, `validacao` e `teste`.
3. Crie o `data/custom.yaml` com os caminhos das pastas e a lista de classes.
4. Abra o notebook `Detector_Produtos_Higiene_YOLOv7.ipynb` no Google Colab (GPU T4) e execute as células em ordem.

**Observação de compatibilidade:** versões novas do PyTorch (2.6 ou superior) bloqueiam o carregamento dos checkpoints do YOLOv7 (`weights_only=True`). A correção foi forçar `weights_only=False` no `torch.load` do `train.py`. Faça isso apenas com arquivos de confiança.

## Créditos

- Rede YOLOv7: Wang, Bochkovskiy e Liao, *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors* (2022). Repositório: https://github.com/WongKinYiu/yolov7

## Licença

MIT.

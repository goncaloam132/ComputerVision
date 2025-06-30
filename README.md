# Computer Vision - Deteção de Trânsito e Objetos com YOLOv8

Este projeto detecta, conta e anota veículos e outros objetos em imagens usando o modelo YOLOv8. O pipeline realiza inferência, extração de caixas, contagem por classe e salva imagens anotadas, facilitando análises de tráfego e demonstrações de visão computacional.

---

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO
pip install -r requirements.txt
```

**Requisitos:**  
- Python 3.8+
- [ultralytics](https://pypi.org/project/ultralytics/)
- opencv-python
- numpy
- tqdm

---

## Uso

Execute o script principal para detectar objetos em uma ou várias imagens:

```bash
python detect_traffic.py -i imagens/
```

**Argumentos principais:**
- `-i`, `--input`: Caminho(s) da(s) imagem(ns) ou diretório de entrada (obrigatório)
- `-o`, `--output`: Caminho do arquivo de saída da imagem anotada (opcional, padrão: pasta `resultados/`)
- `-m`, `--model`: Caminho do modelo YOLOv8 a ser usado (padrão: yolov8n.pt)
- `--csv`: Caminho do arquivo CSV para salvar as contagens (opcional)
- `--json`: Caminho do arquivo JSON para salvar as contagens (opcional)
- `--show`: Exibe a imagem anotada ao final
- `--only-vehicles`: Conta apenas veículos (car, truck, motorcycle, bus)

**Exemplo de uso:**
```bash
python detect_traffic.py -i imagens/ -o resultados/
```
As imagens anotadas serão salvas na pasta `resultados/` como `output_nomeoriginal.jpg`.

---

## Estrutura do Projeto

```
.
├── detect_traffic.py        # Script principal de inferência e anotação
├── imagens/                 # Imagens de teste para inferência
├── resultados/              # Imagens de saída anotadas pelo modelo
├── assets/                  # (Opcional) Screenshots, GIFs e exemplos visuais
├── yolov8n.pt               # Peso do modelo YOLOv8
├── requirements.txt         # Dependências do projeto
├── README.md                # Documentação principal
└── docs/
    └── approach.md          # Descrição da abordagem e desafios
```

---

## Abordagem & Desafios

O pipeline, principais problemas enfrentados e soluções estão detalhados em [docs/approach.md](docs/approach.md).


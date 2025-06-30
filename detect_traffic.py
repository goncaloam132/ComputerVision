import argparse
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import json
import csv

# Paleta de cores para até 20 classes
COLORS = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(20)]


def load_model(model_path: str):
    """Carrega o modelo YOLOv8."""
    try:
        model = YOLO(model_path)
        names = model.model.names
        return model, names
    except Exception as e:
        sys.exit(f"Erro ao carregar modelo: {e}")


def run_inference(model, img_path: str):
    """Executa a inferência e retorna o resultado do YOLO."""
    results = model.predict(source=img_path, verbose=False)
    return results[0]


def count_classes(res, names, filter_classes=None):
    """Conta instâncias por classe, com filtro opcional."""
    if hasattr(res, 'boxes') and res.boxes is not None and hasattr(res.boxes, 'cls') and res.boxes.cls is not None:
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    else:
        cls_ids = np.array([], dtype=int)
    counts = {}
    for cid in cls_ids:
        label = names[cid]
        if (filter_classes is None) or (label in filter_classes):
            counts[label] = counts.get(label, 0) + 1
    return counts, cls_ids


def draw_boxes(img, res, names, filter_classes=None):
    """Desenha bounding boxes e legendas na imagem, mesmo que res seja um tensor."""
    # Tentamos primeiro UX com Results.boxes
    if hasattr(res, 'boxes') and res.boxes is not None:
        boxes   = res.boxes.xyxy.cpu().numpy()
        confs   = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    else:
        # fallback do tensor bruto Nx6 ([x1,y1,x2,y2,conf,cls])
        arr = res.cpu().numpy() if hasattr(res, 'cpu') else res
        if arr.ndim == 2 and arr.shape[1] >= 6:
            boxes   = arr[:, :4]
            confs   = arr[:, 4]
            cls_ids = arr[:, 5].astype(int)
        else:
            # nada para desenhar
            return img

    for box, cid, conf in zip(boxes, cls_ids, confs):
        label = names[cid]
        if filter_classes and label not in filter_classes:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        text = f"{label} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # desenha fundo do texto para garantir visibilidade
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img



def overlay_counts(img, counts):
    """Sobrepõe o resumo de contagem na imagem, no canto superior esquerdo, em azul."""
    y0 = 30
    for label, cnt in counts.items():
        text = f"{label}: {cnt}"
        cv2.putText(img, text, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y0 += 30
    return img


def save_counts_csv(counts_dict, out_csv):
    """Salva as contagens em CSV."""
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['classe', 'contagem'])
        for label, cnt in counts_dict.items():
            writer.writerow([label, cnt])


def save_counts_json(counts_dict, out_json):
    """Salva as contagens em JSON."""
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(counts_dict, f, ensure_ascii=False, indent=2)


def process_image(img_path, model, names, args, filter_classes=None):
    """Processa uma imagem: inferência, anotação manual, contagem e exportação."""
    if not os.path.exists(img_path):
        print(f"[ERRO] Imagem não encontrada: {img_path}")
        return None

    # 1) Inferência
    res = model.predict(source=img_path, verbose=False)[0]

    # 2) Contagem de instâncias
    if hasattr(res, 'boxes') and res.boxes is not None and res.boxes.cls is not None:
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    else:
        # fallback para tensor bruto (N×6 ou N×7)
        arr = res.cpu().numpy() if hasattr(res, 'cpu') else res
        cls_ids = arr[:, 5].astype(int) if arr.ndim == 2 and arr.shape[1] >= 6 else np.array([], int)
    counts = {}
    for cid in cls_ids:
        label = names[cid]
        if (filter_classes is None) or (label in filter_classes):
            counts[label] = counts.get(label, 0) + 1

    # 3) Leitura da imagem original
    img = cv2.imread(img_path)

    # 4) Desenha as caixas e legendas
    img = draw_boxes(img, res, names, filter_classes)

    # 5) Sobrepõe o resumo de contagens
    img = overlay_counts(img, counts)

    # 6) Salva imagem anotada
    output_dir = "resultados"
    os.makedirs(output_dir, exist_ok=True)
    out_img = args.output if args.output else os.path.join(output_dir, f"output_{os.path.basename(img_path)}")
    cv2.imwrite(out_img, img)

    # 7) Exporta CSV/JSON se solicitado
    if args.csv:
        save_counts_csv(counts, args.csv)
    if args.json:
        save_counts_json(counts, args.json)

    # 8) Exibe na tela (opcional)
    if args.show:
        cv2.imshow('Detecção', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Contagens para {img_path}: {counts}")
    print(f"Resultado salvo em {out_img}")
    return counts



def main():
    parser = argparse.ArgumentParser(description='Contagem de veículos com YOLOv8')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='Imagem ou diretório de entrada')
    parser.add_argument('-o', '--output', type=str, help='Arquivo de saída da imagem anotada')
    parser.add_argument('-m', '--model', type=str, default='yolov8n.pt', help='Modelo YOLOv8 a ser usado')
    parser.add_argument('--csv', type=str, help='Arquivo CSV para salvar contagens')
    parser.add_argument('--json', type=str, help='Arquivo JSON para salvar contagens')
    parser.add_argument('--show', action='store_true', help='Exibir imagem anotada ao final')
    parser.add_argument('--only-vehicles', action='store_true', help='Contar apenas veículos (car, truck, motorcycle, bus)')
    args = parser.parse_args()

    # Classes de veículos do COCO
    vehicle_classes = {'car', 'truck', 'motorcycle', 'bus'}
    filter_classes = vehicle_classes if args.only_vehicles else None

    # Carrega modelo
    model, names = load_model(args.model)

    # Lida com múltiplas imagens ou diretório
    img_list = []
    for inp in args.input:
        if os.path.isdir(inp):
            img_list.extend([os.path.join(inp, f) for f in os.listdir(inp) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            img_list.append(inp)

    all_counts = {}
    for img_path in tqdm(img_list, desc='Processando imagens'):
        counts = process_image(img_path, model, names, args, filter_classes)
        if counts is not None:
            all_counts[os.path.basename(img_path)] = counts

    # Exporta CSV/JSON geral se múltiplas imagens
    if len(img_list) > 1:
        if args.csv:
            with open(args.csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['imagem', 'classe', 'contagem'])
                for img_name, counts in all_counts.items():
                    for label, cnt in counts.items():
                        writer.writerow([img_name, label, cnt])
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(all_counts, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

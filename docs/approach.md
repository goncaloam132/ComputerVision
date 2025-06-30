# Abordagem e Desafios

## Pipeline do Projeto

1. **Inferência**: O script carrega o modelo YOLOv8 e executa a inferência sobre as imagens do diretório de entrada.
2. **Extração de Caixas**: As detecções são extraídas do objeto de resultados retornado pelo modelo.
3. **Contagem**: O script conta o número de objetos detectados por classe (carro, caminhão, moto, etc.).
4. **Anotação**: As imagens são salvas com bounding boxes e labels desenhados sobre os objetos detectados.

## Principais Desafios

- **Conversão de formatos (tensor vs. Results)**: O objeto retornado pelo YOLOv8 pode ser um tensor ou um objeto Results, dependendo da versão da biblioteca. Solução: uso de métodos de acesso compatíveis e checagem de tipo.
- **Caminhos de diretório**: Garantir que os caminhos de entrada e saída funcionem em diferentes sistemas operacionais. Solução: uso do módulo `os` para manipulação de paths.
- **Performance**: Processamento em lote pode ser lento em CPU. Solução: uso de GPU se disponível e barra de progresso com tqdm.
- **Visualização**: Garantir que as anotações fiquem legíveis e salvas corretamente. Solução: uso do OpenCV para desenhar bounding boxes e labels.

## Possíveis Melhorias
- Adicionar suporte a vídeos.
- Exportar resultados em outros formatos (CSV, JSON).
- Interface gráfica para facilitar o uso. 
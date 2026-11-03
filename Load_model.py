from huggingface_hub import sync_bucket

from huggingface_hub import download_bucket_files
from pathlib import Path
import os
from huggingface_hub import HfApi, list_bucket_tree
from datasets import load_dataset as HF_load_dataset
from datasets import  Image


os.environ["HF_TOKEN"] = "mi_token"

bucket_id = "z72pepee/tfm-explicabilidad"
remote_path = "Modelos_Entrenados"

token= "mi_token"
api = HfApi(token=token)


RUTAS_MODELOS = {
    "dataset1_microsoft": "Modelos_Entrenados/experimento_dataset1_microsoft/",
    "dataset2_microsoft":  "Modelos_Entrenados/experimento_dataset2_microsoft/",
    "dataset3_microsoft":  "Modelos_Entrenados/experimento_dataset3_microsoft/",
    "dataset1_googleViT":   "Modelos_Entrenados/experimento_dataset1_googleViT/",
    "dataset2_googleViT":  "Modelos_Entrenados/experimento_dataset2_googleViT/",
    "dataset3_googleViT":  "Modelos_Entrenados/experimento_dataset3_googleViT/",
    "dataset1_facebook_dinov2":   "Modelos_Entrenados/experimento_dataset1_facebook_dinov2/",
    "dataset2_facebook_dinov2":    "Modelos_Entrenados/experimento_dataset2_facebook_dinov2/",
    "dataset3_facebook_dinov2":    "Modelos_Entrenados/experimento_dataset3_facebook_dinov2/",
    "dataset3_facebook_dinov2_small_destilado":    "Modelos_Entrenados/experimento_dataset3_facebook_dinov2_SMALL_DESTILADO/"
}

import glob
import os
import re
from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_model(nombre_modelo,
    output_attentions=False, output_hidden_states=False):
    """
    Busca automáticamente la carpeta que empieza por 'checkpoint' dentro de la ruta
    de ese modelo y carga el procesador y el modelo ViT de clasificación de imágenes.

    Args:
        nombre_modelo (str): El nombre del modelo que usaremos para llegar donde está el dataset/modelo.

    Returns:
        tuple: (processor, model) cargados correctamente.
    """

    if nombre_modelo not in RUTAS_MODELOS:
        raise ValueError(f" El modelo '{nombre_modelo}' no está configurado.")

    ruta_base = RUTAS_MODELOS[nombre_modelo]

    # Obtenemos los elementos de la ruta base directamente desde el bucket
    elementos_bucket = list_bucket_tree(bucket_id, prefix=ruta_base)
    
    # Usamos un 'set' para guardar las rutas de las carpetas y evitar duplicados automáticamente
    carpetas_checkpoint = set()
    
    for item in elementos_bucket:
        if item.type == "file":
            # Extraemos la ruta de la carpeta donde está el archivo
            directorio_padre = os.path.dirname(item.path)
            
            # Si el nombre final de esa carpeta contiene 'checkpoint', la guardamos
            if "checkpoint" in os.path.basename(directorio_padre):
                carpetas_checkpoint.add(directorio_padre)

    # Convertimos el set de vuelta a lista
    resultados = list(carpetas_checkpoint)
    # Control de errores por si alguna ruta está mal escrita
    if not resultados:
        # Si no hay checkpoints, asumimos que los archivos están en la carpeta raíz
        print(f" No se encontraron carpetas 'checkpoint'. Se cargará el modelo directamente desde la raíz: {ruta_base}")
        ruta_exacta = ruta_base
    else:
        # Función auxiliar para extraer el número del nombre y ordenarlos correctamente
        def obtener_numero_checkpoint(ruta):
            match = re.search(r'checkpoint-(\d+)', os.path.basename(ruta))
            return int(match.group(1)) if match else 0

        # Ordenamos los resultados numéricamente
        resultados = sorted(resultados, key=obtener_numero_checkpoint)
        cantidad = len(resultados)

        # Extraemos la ruta exacta según la cantidad de checkpoints
        if cantidad == 1:
            ruta_exacta = resultados[0]
        elif cantidad >= 6:
            ruta_exacta = resultados[-6] # Toma el sexto empezando por el final
        else:
            ruta_exacta = resultados[-1] # Fallback si hay entre 2 y 5

    print(f" Carpeta detectada: {os.path.basename(ruta_exacta)}")
    print(f" Cargando procesador y modelo desde: {ruta_exacta}...")

    archivos_remotos = [
    item.path
    for item in list_bucket_tree(bucket_id, prefix=ruta_exacta)
    if item.type == "file"
    ]

    # Construimos la lista de tuplas que necesita download_bucket_files
    # Formato: (ruta_en_el_bucket, ruta_en_tu_disco_local)
    # Al poner (ruta, ruta) se guardarán conservando la misma estructura de carpetas.
    archivos_a_descargar = [
        (ruta, ruta)
        for ruta in archivos_remotos
    ]


    # Descargamos los archivos en bloque
    if archivos_a_descargar:
        download_bucket_files(
            bucket_id,
            files=archivos_a_descargar
        )
        print("¡Descarga completada con éxito!")
    else:
        print("No se encontró nada para descargar (la carpeta no existe o está vacía).")

    # Cargamos los componentes de Hugging Face
    hf_kwargs = {}
    if output_attentions:
        hf_kwargs["output_attentions"] = True
    if output_hidden_states:
        hf_kwargs["output_hidden_states"] = True

    # Cargamos el procesador y el modelo pasando los argumentos empaquetados (**hf_kwargs)
    processor = AutoImageProcessor.from_pretrained(ruta_exacta)
    model = AutoModelForImageClassification.from_pretrained(ruta_exacta, **hf_kwargs)

    print(" Carga en memoria completada con éxito")
    return processor, model


def load_dataset(nombre_dataset):
  """
    Busca y carga el dataset con el nombre correspondiente
    Args:
        nombre_dataset(str): El nombre del dataset que cargaremos

    Returns:
        data: Datos cargados correctamente.
    """
  data = HF_load_dataset("z72pepee/skin-gan-review-baseline", nombre_dataset, token="mi_token")
  return data
    
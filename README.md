# Experimentación y Desarrollo - TFM

Este es el repositorio de trabajo y experimentación de mi Trabajo de Fin de Máster (TFM). Aquí se recogen todas las pruebas, análisis y ejecuciones prácticas que sustentan la memoria del proyecto.

## Relación con el repositorio público

Como parte fundamental de este TFM, he desarrollado una librería llamada **GlassBox_ViT**. Esa librería es de código abierto y tiene su propio repositorio con su código fuente, arquitectura y ejemplos de uso limpios.

Puedes consultar la librería oficial aquí:
[Repositorio de Glassbox_ViT](https://github.com/oscninrob/GlassBox-ViT)

**¿Qué diferencia hay con este repositorio?**
Mientras que `glassbox_vit` es el producto principal empaquetado, este repositorio es mi entorno de pruebas real para el TFM. Aquí uso las funciones y clases de  mi librería, pero también uso código de terceros, otras herramientas y procesos de datos que no pertenecen a la librería, como el entrenamiento de los modelos o el uso de LLM's, pero que son necesarios para sacar las conclusiones de mi trabajo.

## Contenido del repositorio

El código de este repositorio refleja el flujo de trabajo de mis experimentos y está dividido en las siguientes áreas principales:

*   **Entrenamiento de modelos:** Scripts y cuadernos dedicados al entrenamiento de los modelos predictivos base.
*   **Destilación (Distilling):** Implementación donde utilizo mi librería (`glassbox_vit`) para generar un nuevo modelo mediante técnicas de destilación de conocimiento.
*   **Explicabilidad (XAI) y análisis del aprendizaje:** Generación de imágenes de explicabilidad utilizando métodos implementados en `glassbox_vit`. Además, se por otro lado usamos TracIn para extraer conclusiones sobre el proceso de aprendizaje de los modelos.
*   **Evaluación mediante LLMs:** Un notebook específico encargado de realizar llamadas a diferentes modelos de lenguaje (LLMs). Su función es actuar como jueces para evaluar y puntuar las imágenes de explicabilidad generadas anteriormente.
*   **Análisis de resultados:** Archivos y cuadernos donde cruzo todos los datos para extraer conclusiones para el TFM. Esto incluye tanto el análisis de las métricas puras de los modelos ya entrenados, como el estudio de las puntuaciones devueltas por los LLMs.
*   **Archivos auxiliares:** Funciones de apoyo, utilidades de código y configuraciones necesarias para que todo el pipeline de experimentación funcione correctamente.

---
*Repositorio de trabajo personal para la defensa del Trabajo de Fin de Máster.*

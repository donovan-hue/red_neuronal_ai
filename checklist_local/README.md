# Checklist local de auditoría (74 puntos)

Inicia el panel desde la raíz del proyecto:

```bash
python3 checklist_local/server.py
```

Abre `http://localhost:8765` y pulsa **Ejecutar verificaciones locales**. Cada casilla cambia únicamente cuando su prueba o comprobación termina. Los resultados se guardan localmente en `checklist_local/progress.json`; no contienen preguntas ni archivos del usuario.

El panel no envía consultas, mensajes, capturas, fotos ni artefactos a servicios externos. No instala dependencias ni descarga el modelo. Los controles que requieren `torch` y `tokenizers` quedan como bloqueados si esos paquetes no están presentes.

# Lista única de desarrollo pendiente

**Objetivo:** NEXUS para uso individual y confidencial en Android/Termux, con
KRONOS MCP local y supervisión del propietario. Esta es la lista de trabajo para
continuar en otras sesiones; `checklist_local/` se conserva como panel de
verificación técnica, no como hoja de ruta.

**Cómo mantenerla:** trabajar por bloques, alternando tareas complejas y sencillas
cuando sea práctico. Eliminar cada casilla únicamente después de implementar,
probar y documentar el resultado. Registrar evidencia en el commit o PR. No
considerar terminado el proyecto mientras queden casillas aplicables abiertas;
las tareas condicionadas requieren una decisión explícita del propietario.
No subir secretos ni datos privados al repositorio.

**Estado al crear esta lista:** hay avances en seguridad, autenticación y modo
privado. La integración con KRONOS y el funcionamiento en Termux no se han probado
desde NEXUS. Las pruebas locales no certifican Android ni seguridad absoluta.
El modelo local exige un checkpoint y dependencias compatibles.

## 1. Seguridad, privacidad y control local

- [ ] Revisar y probar de extremo a extremo la autenticación individual, los roles,
      la caducidad, la rotación, la revocación y los fallos cerrados.
- [ ] Probar el modo privado contra conexiones de red externas, selección accidental
      de proveedor y uso de credenciales de otras aplicaciones del dispositivo.
- [ ] Revisar y reforzar la protección de datos, observaciones y documentos contra
      prompt injection; ejecutar pruebas adversarias y documentar sus límites.
- [ ] Inventariar los datos persistidos, logs, bases SQLite, checkpoints y archivos
      temporales; definir y probar una política de retención y borrado.
- [ ] Diseñar y probar respaldo y restauración privada sin incluir secretos en Git.
- [ ] Evaluar e implementar cifrado de archivos o respaldo si el propietario lo
      requiere; documentar dependencia del cifrado y bloqueo de Android.
- [ ] Revisar protección de claves, permisos de archivos y exposición de información
      en logs, interfaz y mensajes de error.
- [ ] Validar límites de solicitudes, concurrencia y cuota de llamadas bajo carga;
      documentar que la cuota actual cuenta llamadas y no dinero.
- [ ] Mantener Python arbitrario deshabilitado; habilitarlo solo si el propietario
      lo solicita y existe un sandbox real, independiente y probado.

## 2. Android, Termux y operación individual

- [ ] Instalar en el teléfono real y registrar versión de Android, arquitectura,
      versión de Termux, almacenamiento, memoria y versión de Python.
- [ ] Verificar disponibilidad e instalación de NumPy, tokenizers y PyTorch en el
      dispositivo; resolver o documentar una alternativa si son incompatibles.
- [ ] Descargar y verificar el checkpoint y tokenizer compatibles; ejecutar el
      modelo local y medir memoria, latencia, batería y temperatura.
- [ ] Probar inicio, autenticación, conversación, cierre y reinicio del modo privado
      desde Termux y navegador móvil, sin exponer servicios a la red local.
- [ ] Probar funcionamiento sin internet una vez instalados los artefactos y
      confirmar que el modelo no realiza llamadas a proveedores externos.
- [ ] Manejar errores de instalación, falta de checkpoint y recursos insuficientes
      con instrucciones comprensibles en lugar de fallos silenciosos.
- [ ] Crear una guía de instalación, operación, actualización y recuperación en
      Termux sin publicar claves ni asumir una distribución de paquetes funcional.

## 3. Integración NEXUS ↔ KRONOS MCP

- [ ] Diseñar un cliente MCP compatible con `donovan-hue/KRONOS_MCP` y el transporte
      local elegido; definir ciclo de vida, reconexión y cierre de procesos.
- [ ] Descubrir y registrar herramientas MCP con esquemas validados y una lista
      explícita de herramientas autorizadas para el propietario.
- [ ] Implementar llamadas y respuestas con límites de tiempo, tamaño y errores;
      tratar la salida MCP como contenido no confiable.
- [ ] Aislar tokens y procesos; distinguir herramientas locales de las que acceden
      al backend y exigir autorización explícita para usar internet.
- [ ] Probar desconexión, permisos, entradas inválidas y salidas maliciosas sin
      filtrar secretos ni afirmar que operaciones no realizadas se ejecutaron.
- [ ] Probar en Termux la conexión completa, incluyendo una herramienta local y,
      solo si el propietario lo autoriza, una herramienta del backend KRONOS.
- [ ] Documentar instalación conjunta, requisitos, configuración privada y cierre.

## 4. API y arquitectura

- [ ] Versionar y documentar las APIs mediante OpenAPI y ejemplos verificables.
- [ ] Unificar validación y formato seguro de errores en las rutas relevantes.
- [ ] Añadir cancelación y seguimiento persistente de tareas; cubrir fallos,
      reinicio y limpieza de procesos hijos.
- [ ] Resolver sincronización de modelos entre procesos o imponer y documentar
      operativamente un único worker para entrenamiento de la API clásica.
- [ ] Separar configuración por entorno y validar valores inválidos al iniciar.
- [ ] Revisar código duplicado entre NEXUS y `nexus_single.py` y consolidarlo sin
      romper la demostración independiente.

## 5. Red neuronal clásica

- [ ] Añadir división reproducible de entrenamiento y validación, evitando fugas.
- [ ] Versionar checkpoints y proporcionar recuperación segura de versiones.
- [ ] Permitir configuración validada de arquitectura y funciones de activación.
- [ ] Ampliar pruebas de entrenamiento, persistencia, fallos y simultaneidad.

## 6. Agente, conversación e interfaz

- [ ] Conectar la demostración del avatar con el agente real sin afirmar que la
      simulación actual ya está conectada.
- [ ] Implementar historial persistente opcional y administración de conversaciones,
      aislados por identidad y sujetos a retención y eliminación.
- [ ] Implementar cancelación y streaming real de generación; distinguirlo del
      envío actual de fragmentos después de generar la respuesta.
- [ ] Añadir registro y administración segura de herramientas y sus permisos.
- [ ] Añadir presupuestos de tiempo y trabajo por tarea y recuperación de errores.
- [ ] Añadir enrutamiento entre modelos y seguimiento de tokens/coste si se decide
      usar proveedores; no confundir la cuota de llamadas con un límite monetario.
- [ ] Añadir reintentos controlados y exportación privada de conversaciones.

## 7. RAG y memoria

- [ ] Incorporar carga, administración, eliminación e indexación de documentos.
- [ ] Soportar PDF y HTML con límites de tamaño y manejo seguro de entradas.
- [ ] Proporcionar citas y fragmentos verificables asociados a la respuesta.
- [ ] Evaluar e implementar búsqueda híbrida semántica/BM25 y reordenamiento.
- [ ] Actualizar y eliminar selectivamente documentos e índices durante ejecución.
- [ ] Implementar memoria persistente opcional aislada por identidad, con consulta,
      edición, eliminación, caducidad y límites; no compartir datos entre usuarios.

## 8. Datos, entrenamiento y evaluación

- [ ] Ampliar y limpiar el corpus respetando licencias y confidencialidad.
- [ ] Detectar duplicados y contaminación entre datos de entrenamiento y prueba.
- [ ] Reforzar checkpoints reanudables, recuperación de fallos y registro de
      experimentos; comprobar que el mejor modelo de validación se conserva.
- [ ] Ampliar evaluaciones en español y por dominio con conjuntos reproducibles.
- [ ] Añadir benchmarks externos cuando sean aplicables y comparar versiones.
- [ ] Medir calidad, latencia, uso de memoria y, cuando aplique, coste por tarea.
- [ ] Ejecutar evaluación de seguridad, inyección y regresiones antes de cada release.

## 9. Calidad, mantenimiento y documentación

- [ ] Ampliar pruebas unitarias, integración, extremo a extremo, carga y concurrencia.
- [ ] Comprobar la integración continua en GitHub y ampliar análisis estático.
- [ ] Fijar o bloquear versiones de dependencias, revisar compatibilidad y actualizar
      de forma controlada, especialmente en Termux.
- [ ] Añadir análisis automático de vulnerabilidades y controles de secretos.
- [ ] Añadir registros estructurados, métricas y alertas sin almacenar datos sensibles.
- [ ] Actualizar documentación y ejemplos tras cada cambio; mantener esta como la
      única lista de desarrollo pendiente.

## 10. Antes de fusionar y cerrar

- [ ] Revisar que todas las casillas aplicables de los bloques anteriores se hayan
      resuelto; registrar decisiones explícitas para cualquier tarea condicionada.
- [ ] Revisar el diff y el historial, sin secretos, archivos privados ni artefactos
      grandes; ejecutar pruebas y registrar las limitaciones que no puedan probarse.
- [ ] Ejecutar la suite y el panel de auditoría local; revisar los resultados de CI.
- [ ] Verificar el flujo completo en el teléfono real con NEXUS y KRONOS MCP.
- [ ] Actualizar y subir los cambios a la rama de la sesión; abrir o actualizar un
      PR sin fusionarlo hasta contar con autorización explícita del propietario.
- [ ] Con autorización explícita, fusionar el PR, verificar el estado final y
      confirmar que no quedan tareas abiertas aplicables.

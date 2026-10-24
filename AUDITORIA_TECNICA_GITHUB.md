# Auditoría técnica del repositorio

**Fecha:** 2026-10-23
**Alcance principal:** rama GitHub `arena/dd61cae6-red-neuronal-ai`, commit `00d9c8b83c81a80b6a58da95e3e7e6722a451c7d`.
**Alcance secundario:** árbol local del checkout; se reporta por separado y no se atribuye al commit remoto.

## Resumen ejecutivo

El commit remoto tiene una suite pequeña que pasa, pero eso no acredita que NEXUS completo esté listo: en este entorno no estaban disponibles `torch` ni `tokenizers`, no había claves de proveedor y no se hicieron llamadas a APIs. La revisión sí encontró defectos reproducibles en la generación de datos y en varias rutas de código, además de riesgos condicionados al despliegue.

**Lo probado contra el commit remoto:**

- `python -m unittest discover -s tests -v`: **16/16 OK** en un archivo extraído del commit remoto.
- `python3 nexus_single.py selftest`: **84/84**. Es el auto-test determinista del servidor autocontenido; **no** prueba las integraciones del runtime NEXUS.
- `compileall` y `bash -n`: finalizaron correctamente.
- No fue posible ejecutar el runtime NEXUS íntegro: faltan `torch` y `tokenizers`. Tampoco se probaron llamadas reales a OpenAI, Anthropic o Gemini: no había credenciales.
- `git diff --check` no se puede aplicar al archivo extraído sin `.git`; sí pasó en el checkout local.

**Checkout local, separado del resultado de GitHub:** en el momento de la auditoría local, HEAD era `63b2d1a` y había cambios locales y archivos no rastreados respecto de ese commit. El ref remoto auditado está en `00d9c8b`. La ejecución más reciente del checklist local, iniciada con `.venv/bin/python`, quedó en **72 aprobados, 1 fallido y 1 bloqueado**: el punto 59 detecta el traceback SSE y el runtime NEXUS sigue bloqueado por dependencias. Por separado, la suite local dio **17/17 OK**, y el selftest 84/84; esos números pertenecen al árbol local de entonces, no al commit de GitHub. El árbol auditado incluía un saneamiento parcial del mensaje de error que no estaba en el remoto, pero **todavía transmitía el campo `tb`**. En el momento de redactar este informe, no se había hecho commit ni push.

## Hallazgos confirmados

### Seguridad y operación

#### 1. El endpoint SSE devuelve tracebacks internos — **riesgo confirmado en GitHub**

En el commit auditado, el manejador de errores SSE envía tanto el mensaje de excepción como `traceback.format_exc()[-800:]` al cliente: [`nexus/src/server.py`, líneas 296–298](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L296-L298). El servidor autocontenido también devuelve el tipo y texto de excepciones al cliente en [`nexus_single.py`, líneas 523–524](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus_single.py#L523-L524).

**Impacto:** puede revelar rutas locales, detalles de implementación y errores de proveedores. El checklist local comprobó que el campo traceback está en el contrato y marcó el punto 59 como fallido; esto es una comprobación de código, no una prueba de explotación remota.

#### 2. La carga de checkpoints permite deserialización pickle — **riesgo condicional**

[`nexus/src/model.py`, línea 135](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/model.py#L134-L141) usa `torch.load(..., weights_only=False)`. Un checkpoint manipulado podría ejecutar código durante la carga.

**Alcance importante:** no encontré una ruta HTTP que permita a un usuario subir directamente un checkpoint. El riesgo requiere que un operador o proceso ponga un archivo no confiable en la ruta de carga; por eso no lo describo como RCE remoto probado. El script de descarga de releases verifica hashes de artefactos, pero eso no convierte en segura cualquier ruta local alternativa.

#### 3. El motor local no escapa tokens estructurales del contenido — **inyección estructural reproducida**

[`nexus/src/engine.py`, líneas 8–16](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L8-L16) concatena directamente el contenido de mensajes dentro de marcadores como `<|system|>` y `<|end|>`. El motor local tokeniza ese texto como entrada ([líneas 46–54](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L46-L54)). Reproduje que un mensaje de usuario con `\n<|end|>\n<|system|>...` conserva literalmente el marcador `<|system|>` en el prompt renderizado.

**Impacto y límite:** puede alterar la estructura de turnos que recibe el modelo **local**. No es evidencia de que la misma cadena rompa la separación de roles de un proveedor API, que recibe mensajes estructurados de otra forma.

#### 4. La memoria SQLite no aísla usuarios — **riesgo condicionado a despliegue multiusuario**

NEXUS crea una única instancia de memoria en la ruta configurada al arrancar ([`nexus/src/server.py`, líneas 49–53](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L49-L53)); `recall` consulta los hechos de esa base sin un identificador de usuario ([`nexus/src/tools.py`, líneas 181–188](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L181-L188)). `nexus_single.py` también fija una base compartida en `.nexus_memory.sqlite` (línea 27).

**Impacto y límite:** dos usuarios atendidos por la misma instancia podrían recuperar hechos guardados por el otro. Esto es un problema si se despliega como servicio multiusuario; no demuestra exposición en una instalación personal de un solo usuario.

#### 5. El cliente CLI no satisface el contrato de autenticación API y fija el puerto

[`nexus/src/ask.py`, líneas 3–7](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/ask.py#L3-L7) llama siempre a `127.0.0.1:8080` y sólo envía `Content-Type`; no envía `X-Nexus-Api-Token`. El servidor toma el puerto de `PORT` o de `config.json` ([`nexus/src/server.py`, líneas 333–335](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L333-L335)) y exige ese token cuando está usando proveedor API ([líneas 249–257](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L249-L257)).

**Impacto:** el CLI puede funcionar sólo en el puerto por defecto y con motor local; con proveedor API autenticado fallará con 401. No se hizo una llamada de integración porque no había servidor NEXUS ejecutable ni credenciales.

### Datos, evaluación y comportamiento

#### 6. El generador de silogismos puede asignar respuestas incorrectas — **error de datos reproducido**

[`nexus/src/synth.py`, líneas 112–121](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/synth.py#L112-L121) elige la premisa 1 y la regla de la premisa 2 de forma independiente, pero siempre etiqueta la respuesta como `si`. Con semilla 1 reproduje:

> Premisa 1: Las kotas nadan. Premisa 2: todo lo que vuela come metal. … respuesta guardada: `si`.

La propiedad no se deduce de esas premisas. `make_item` usa el generador en la mezcla sintética de entrenamiento. La evaluación predeterminada no incluye `syllogism`, así que no afirmo que sus resultados por defecto estén contaminados por esta tarea; sí afecta los datos sintéticos y cualquier evaluación que la seleccione explícitamente.

#### 7. `synth_share = 0` no desactiva necesariamente los ejemplos sintéticos

[`nexus/src/corpus.py`, líneas 140–145](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/corpus.py#L140-L145) impone `max(2000, cap)` bloques y toma hasta 3.000 ejemplos de herramientas (o hasta `n_synth` si es mayor), según lo que exista en `tools`. Si `synth_samples` produce bloques, configurar la proporción en cero todavía añade ejemplos sintéticos. El archivo permite explícitamente `synth_share = 0` (validación en líneas 132–133), de modo que la opción no cumple la expectativa normal de “cero”.

#### 8. La partición de validación separa párrafos, no libros/orígenes

[`nexus/src/corpus.py`, líneas 28–56](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/corpus.py#L28-L56) divide cada libro en muchos fragmentos; después, líneas 145–153 mezclan y barajan esos fragmentos antes de reservar el 1 % para validación. Por tanto, fragmentos del mismo libro pueden quedar en entrenamiento y validación.

**Impacto:** la pérdida de validación puede ser optimista por similitud de fuente. No afirmo que exista duplicación exacta sin haber calculado similitud del corpus instalado; el defecto confirmado es que la partición no agrupa por libro.

#### 9. `eval.py --tasks ""` termina dividiendo entre cero — **caso reproducible por inspección y helper**

La lista vacía produce un pool vacío (`items(..., [])` devuelve 0 elementos); luego [`nexus/src/eval.py`, líneas 43–47 y 96–101](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/eval.py#L43-L47) calcula `correct / len(pool)` sin rechazar que no haya tareas. Con un motor que llegue a ejecutar la evaluación, la ruta culmina en `ZeroDivisionError`. No lo presento como fallo de la suite existente; esa ruta vacía no está cubierta por ella.

#### 10. El verificador acepta cualquier texto con `PASS` — **falso positivo reproducido**

[`nexus/src/agent.py`, líneas 303–312](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L303-L312) marca éxito si encuentra `PASS` en cualquier parte y no encuentra `REVISA`. Lo probé con una respuesta crítica en español equivalente a “el resultado no es correcto, pero PASS aparece en el texto”; `verify()` devolvió `True`.

**Impacto:** el campo `verify_ok` no es una garantía formal cuando no hay comprobación matemática determinista. La rama con valor de calculadora sí compara el resultado exacto antes de consultar al crítico.

#### 11. Se guardan trayectorias, pero los ejemplos sólo reutilizan pregunta y respuesta

`Memory.good_traces()` obtiene también la columna `steps` ([`nexus/src/tools.py`, líneas 194–206](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L194-L206)), pero el agente construye los ejemplares con `t[1]` y `t[2]` —pregunta y respuesta—, omitiendo `t[3]` ([`nexus/src/agent.py`, líneas 156–160](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L156-L160)).

**Impacto:** la función anunciada como recuperación de trayectorias no reaprovecha los pasos/herramientas; en la práctica ofrece pares Q→A.

#### 12. La opción `cot` y el argumento `why` no afectan `render_train`

[`nexus/src/synth.py`, líneas 180–194](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/synth.py#L180-L194) acepta ambos argumentos, pero no los consulta al formar la salida. Reproduje que `render_train(..., cot=True)` y `render_train(..., cot=False)` devuelven el mismo texto para los demás argumentos iguales. Se alternan plantillas por `i`, pero no se usa el `why` generado.

#### 13. El tokenizer BPE legado daña texto y no está conectado al camino actual

Reproduje en ambos tamaños de vocabulario, 300 y 8192, con la implementación de [`nexus/src/tokenizer.py`, codificación y decodificación en líneas 127–138](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tokenizer.py#L127-L138):

```text
encode/decode("hello world café") → "hello<|system|>world<|system|>café"
```

También `build_chat(tokens_ids, role_ids)` devuelve sin cambios `tokens_ids` e ignora `role_ids` ([`nexus/src/tokenizer.py`, líneas 162–164](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tokenizer.py#L162-L164)). **Límite:** este módulo es legado/no usado por el pipeline NEXUS activo: la construcción del corpus usa el paquete `tokenizers` y la inferencia carga el adaptador `tokenizer_ext.Tok`; no lo presento como causa de los resultados del runtime actual.

#### 14. La herramienta Python está registrada, pero su ejecución está deshabilitada

La descripción entregada al agente la llama “intérprete Python aislado” ([`nexus/src/tools.py`, líneas 215–216](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L215-L216)); `python_exec()` siempre devuelve que la herramienta está deshabilitada ([líneas 146–148](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L146-L148)). La documentación de arquitectura sí reconoce que está deshabilitada por seguridad. Es una función ofrecida en el registro pero no implementada como ejecución; no es una vulnerabilidad de sandbox observada.

#### 15. La salud de la API clásica no verifica que el modelo esté entrenado

[`api.py`, líneas 14–17](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/api.py#L14-L17) crea una red nueva si falta `modelo.npz`, y `/health` devuelve siempre `status: ok` ([líneas 91–93](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/api.py#L91-L93)). Si se arranca sin el artefacto, la disponibilidad HTTP puede reportarse como sana aunque el modelo usado sea la inicialización nueva y no el entrenado. El artefacto sí está presente en el release revisado; el hallazgo es el comportamiento condicional si falta.

### Funciones descritas, conexiones y documentación

#### 16. El SSE no equivale a generación en streaming del proveedor

`APIEngine.stream()` primero llama `complete()` y sólo después divide el texto recibido en fragmentos ([`nexus/src/engine.py`, líneas 173–176](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L173-L176)); el servidor también lo indica en un comentario ([`nexus/src/server.py`, líneas 285–287](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L285-L287)). La conexión HTTP/SSE existe, pero el texto de respuesta del proveedor no llega token a token: puede haber una espera hasta completar la respuesta.

#### 17. No hay enrutado por coste en `make_engine`

La documentación del roadmap atribuye “enrutado por coste” a `engine.make_engine` ([`nexus/docs/ROADMAP.md`, líneas 74–75](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ROADMAP.md#L74-L75)). El código elige el proveedor API cuya clave disponible aparece primero en la configuración, o el motor local si no encuentra clave; no calcula coste por tarea ni escala según la duda del verificador ([`nexus/src/engine.py`, líneas 98–105 y 180–189](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L98-L105)). Es una discrepancia entre documentación y funcionalidad.

#### 18. La interfaz afirma siempre que hay un transformer entrenado

El mensaje inicial de [`nexus/web/index.html`, línea 62](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/web/index.html#L62) dice “transformer entrenado en este sandbox”; sin embargo, el motor `auto` puede seleccionar un proveedor API según las claves disponibles, y el motor local requiere checkpoint ([`nexus/src/engine.py`, líneas 180–189](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L180-L189)). El texto de bienvenida no refleja el motor realmente activo.

#### 19. Infraestructura e integraciones no verificadas en ejecución

- **Proyecto clásico:** Flask/NumPy y `modelo.npz` local.
- **NEXUS:** servidor HTTP propio y cliente web del mismo proyecto; adaptadores HTTP para OpenAI, Anthropic y Gemini; claves desde variables de entorno. No se hizo ninguna llamada real a esos servicios.
- **Motor local:** PyTorch + `tokenizers` + checkpoint/tokenizer; no arrancó en este entorno por dependencias ausentes.
- **Persistencia y recuperación:** SQLite local para memoria y BM25 local para búsqueda; no encontré un vector store ni una base de datos externa implementados.
- **Datos y artefactos:** scripts descargan textos de Project Gutenberg y assets de releases de GitHub; el script de release comprueba hashes.
- **Tareas administrativas:** entrenamiento y evaluación se lanzan como subprocesos locales con token administrativo.
- **CI:** el commit auditado no contiene archivos en `.github/workflows/`; no hay workflow de GitHub Actions que ejecute estas validaciones automáticamente.

## Conclusión

Hay evidencia suficiente para afirmar que el commit remoto pasa sus pruebas unitarias disponibles y que contiene los problemas funcionales y de seguridad condicionales descritos arriba. No hay evidencia para afirmar que el runtime NEXUS completo, los proveedores API o el entrenamiento funcionen de extremo a extremo en este entorno: esas partes quedaron sin ejecutar. El checklist local refleja correctamente el traceback SSE como fallo y las dependencias faltantes como bloqueo.

Este informe se creó inicialmente como artefacto del workspace local; esta versión se incluye en la pull request para revisión.

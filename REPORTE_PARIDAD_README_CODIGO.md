# Reporte de paridad entre README, descripciones y código

**Repositorio auditado:** GitHub, `arena/dd61cae6-red-neuronal-ai`, commit `00d9c8b83c81a80b6a58da95e3e7e6722a451c7d`.
**Propósito de este reporte:** identificar funciones, opciones y secciones descritas que están incompletas, se comportan distinto o son sólo propuestas. No es una lista de mejoras hipotéticas. Los hallazgos son estáticos salvo donde indico una reproducción.

## Resumen directo

Hay varias descripciones que sí están respaldadas, pero también hay discrepancias concretas entre texto y comportamiento: `tools+router` no enruta con un proveedor API; la frase del roadmap sobre enrutado por coste no corresponde a `make_engine`; la descripción de la herramienta Python promete un intérprete aislado aunque la función siempre responde “deshabilitada”; y la interfaz afirma que hay un transformer entrenado aunque el motor puede ser una API.

Además, existen funciones stub o parcialmente conectadas (`build_chat`, parámetros `cot`/`why`, ejemplos de trayectorias sin los pasos) y capacidades que aparecen en la hoja de ruta pero no están implementadas. Las enumero separadas para no confundir planes con funciones actuales.

## 1. Descripciones que no corresponden plenamente con el comportamiento

### 1.1 `tools+router`: no mide enrutado cuando el motor es un proveedor API

El README anuncia `tools+router` como uno de los brazos del benchmark ([`nexus/README.md`, líneas 27–31](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/README.md#L27-L31)). `eval.py` entrega `router=False` al brazo `tools` y `router=True` al brazo `tools+router` ([líneas 85–87](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/eval.py#L85-L87)).

Pero el motor API implementa `choice()` devolviendo siempre `[]`, porque esos proveedores no entregan puntuaciones comparables ([`nexus/src/engine.py`, líneas 169–171](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L169-L171)). El agente sólo ejecuta la ruta forzada si recibió esa preferencia ([`nexus/src/agent.py`, líneas 207–224](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L207-L224)).

**Resultado preciso:** con un proveedor API, `tools+router` no realiza su decisión de enrutado por log-probabilidad. El brazo aún activa el `guard` determinista al final, mientras `tools` lo desactiva ([`agent.py`, líneas 315–335](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L315-L335)). Por eso, en esa configuración, la comparación mide principalmente el efecto del guard, no el del router que sugiere el nombre. En el motor local, `choice()` sí calcula puntuaciones y el router puede operar.

### 1.2 El enrutado por coste atribuido a `make_engine` no está implementado

La hoja de ruta dice que verificación, autoconsistencia, RAG y “enrutado por coste” ya están esbozados en `engine.make_engine` ([`nexus/docs/ROADMAP.md`, líneas 74–75](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ROADMAP.md#L74-L75)). El README de NEXUS aclara que la hoja de ruta no representa capacidades implementadas ([`nexus/README.md`, línea 48](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/README.md#L48)); aun con esa salvedad, esa frase concreta atribuye el enrutado al código existente.

En realidad `make_engine()` elige el proveedor forzado/configurado; en `auto`, `APIEngine` toma la primera clave presente siguiendo el orden del objeto de configuración ([`nexus/src/engine.py`, líneas 98–105](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L98-L105)); si no encuentra ninguna, `make_engine()` intenta el motor local ([líneas 180–189](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L180-L189)). No compara precios, tarea, latencia ni confianza. En la configuración actual el orden es OpenAI, Anthropic, Gemini ([`nexus/config.json`, líneas 5–23](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/config.json#L5-L23)).

**Conclusión:** hay selección por disponibilidad/configuración; no hay enrutador por coste ni estrategia de “barato primero, escalar si duda”.

### 1.3 La descripción del registro promete Python aislado, pero no ejecuta Python

El registro que ve el agente describe la herramienta `python` como “intérprete Python aislado” ([`nexus/src/tools.py`, líneas 213–215](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L213-L215)). Sin embargo, `python_exec()` siempre devuelve que está deshabilitada ([líneas 146–148](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L146-L148)).

El README y la arquitectura **sí** advierten correctamente que Python está deshabilitado por seguridad ([`nexus/README.md`, líneas 43–45](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/README.md#L43-L45), [`ARCHITECTURE.md`, líneas 18–20](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ARCHITECTURE.md#L18-L20)). La inconsistencia está en la descripción interna que se presenta al modelo. La herramienta es un stub intencional, no una capacidad funcional.

### 1.4 La interfaz afirma que el transformer está entrenado aunque puede no ser el motor activo

El mensaje inicial de la UI dice “transformer entrenado en este sandbox” ([`nexus/web/index.html`, línea 62](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/web/index.html#L62)). El motor `auto` selecciona una API si encuentra una clave disponible; sólo intenta el motor local si no hay proveedor API utilizable ([`nexus/src/engine.py`, líneas 180–189](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L180-L189)). Además, el motor local requiere un checkpoint existente ([`engine.py`, líneas 28–39](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L28-L39)).

**Resultado:** el texto de bienvenida no es una descripción fiable del motor activo. Puede mostrar esa afirmación cuando la sesión usa un proveedor API.

### 1.5 `ask.py` expone una integración CLI que no cumple el contrato de acceso del servidor

El punto de entrada `./run.sh ask` figura en la ayuda del script ([`nexus/run.sh`, líneas 5–8 y 88–92](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/run.sh#L5-L8)). El cliente [`nexus/src/ask.py`, líneas 3–7](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/ask.py#L3-L7) fija `127.0.0.1:8080` y no envía `X-Nexus-Api-Token`. El servidor lee el puerto de `PORT`/config ([`server.py`, líneas 333–335](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L333-L335)) y exige ese header cuando el motor es API ([líneas 249–257](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L249-L257)).

**Resultado:** con motor local y puerto por defecto el contrato puede coincidir; con proveedor API, el CLI no está integrado con la autenticación documentada y recibe rechazo. Con puerto personalizado, apunta al puerto equivocado.

### 1.6 “Trayectorias” se guarda, pero los ejemplos que llegan al prompt son Q→A

La arquitectura dice que la memoria distingue hechos y trayectorias, y que las trayectorias verificadas se usan como ejemplos ([`ARCHITECTURE.md`, línea 26](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ARCHITECTURE.md#L24-L26)). `good_traces()` recupera también `steps`, pero el agente sólo incluye `t[1]`/`t[2]` —pregunta y respuesta—, no `t[3]`, los pasos ([`tools.py`, líneas 194–206](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L194-L206), [`agent.py`, líneas 156–160](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L156-L160)).

**Resultado:** sí reutiliza pares de ejemplos marcados correctos; no reutiliza la trayectoria de razonamiento/herramientas que sugiere el nombre.

### 1.7 El `render_train()` recibe parámetros de razonamiento que no usa

`render_train(q, a, cot=True, why=None, ...)` acepta `cot` y `why`, pero su cuerpo no consulta ninguno ([`nexus/src/synth.py`, líneas 180–194](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/synth.py#L180-L194)). `make_item()` genera un `why` y `corpus.py` lo pasa a `render_train`; ese argumento se descarta ([`synth.py`, líneas 145–149](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/synth.py#L145-L149), [`corpus.py`, líneas 78–88](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/corpus.py#L78-L88)).

La salida sí alterna unas plantillas fijas por `i`; no usa el `why` generado ni cambia cuando se pasa `cot=True` frente a `False`. Lo reproduje con iguales argumentos salvo `cot`.

### 1.8 La bienvenida promete un modelo; `health` del sistema clásico tampoco comprueba readiness

En NEXUS, la UI no se adapta al motor activo (hallazgo 1.4). En el demostrador clásico, si falta `modelo.npz`, `api.py` construye una red nueva y aun así `/health` devuelve siempre `status: ok` ([`api.py`, líneas 14–17 y 91–93](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/api.py#L14-L17)).

**Alcance:** el NPZ está presente en el release revisado; el problema sólo se presenta si el artefacto falta. La ruta de liveness no informa si está sirviendo el modelo entrenado.

### 1.9 El brazo de validación creado por el descargador no es un holdout garantizado por documento

`nexus/README.md` dice que `get_model.sh` crea una partición de validación “no solapada” ([línea 23](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/README.md#L23)). El script sí crea rangos de tokens separados, pero corta el último 1 % del `train.bin` publicado ([`nexus/get_model.sh`, líneas 40–45](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/get_model.sh#L40-L45), y líneas [66–84](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/get_model.sh#L66-L84)). No reconstruye el holdout por libro/documento. La preparación normal divide después de trocear libros en fragmentos y barajarlos ([`corpus.py`, líneas 28–56](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/corpus.py#L28-L56), y líneas [145–153](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/corpus.py#L145-L153)).

**Conclusión matizada:** los rangos de tokens que escribe el script son disjuntos por posición; eso no equivale a un holdout independiente por documento/origen, ni garantiza que los tokens finales no se hayan usado para entrenar el checkpoint publicado. No afirmo haber reproducido una métrica contaminada: la limitación confirmada es el método de partición descrito.

### 1.10 El selector de versión de `get_model.sh` no tiene hashes por versión

El script acepta un tag opcional mediante `VER="${1:-v0.1}"`, pero los tres SHA-256 están fijos a los artefactos de v0.1 ([`nexus/get_model.sh`, líneas 5–9](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/get_model.sh#L5-L9)).

**Resultado:** el valor por defecto v0.1 se puede verificar; pasar otro tag descarga desde ese tag y luego lo compara con hashes de v0.1. Otro release fallará, salvo que sus archivos sean idénticos. El parámetro de versión no es un selector general de releases con sus propios manifiestos.

### 1.11 El nombre `stream` y los eventos SSE no significan streaming del proveedor

`APIEngine.stream()` espera a que termine `complete()` y después divide el texto en trozos ([`engine.py`, líneas 173–176](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/engine.py#L173-L176)). El servidor también resuelve la respuesta antes de enviar los fragmentos finales ([`server.py`, líneas 284–287](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L284-L287)).

**Importante:** la arquitectura actual sí lo advierte expresamente (“no es generación token a token”, [`ARCHITECTURE.md`, línea 13](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ARCHITECTURE.md#L13)). Por tanto, no lo clasifico como contradicción de ese documento; lo clasifico como una capacidad parcial detrás de una interfaz SSE.

## 2. Funciones y módulos incompletos, stubs o fuera del camino activo

### 2.1 `build_chat(tokens_ids, role_ids)` es un stub

Devuelve `tokens_ids` sin cambios e ignora `role_ids` ([`nexus/src/tokenizer.py`, líneas 162–164](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tokenizer.py#L162-L164)). El propio docstring dice “no usado”; por eso es código muerto/incompleto, no una rotura de la ruta NEXUS activa.

### 2.2 El tokenizer BPE legado funciona mal, pero no está conectado a NEXUS actual

La reproducción `encode/decode("hello world café")` devolvió `hello<|system|>world<|system|>café` con vocabularios 300 y 8192. El pipeline activo usa el paquete `tokenizers` y `tokenizer_ext.Tok`, no `BPETokenizer` de este archivo. Es un módulo defectuoso que conviene no confundir con el tokenizer del runtime actual.

### 2.3 El generador de silogismos puede fabricar ejemplos con etiqueta no deducible

En `synth.syllogism()`, las dos premisas se eligen independientemente y la función siempre devuelve `"si"` ([`nexus/src/synth.py`, líneas 112–121](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/synth.py#L112-L121)). Con semilla 1, la premisa “Las kotas nadan” se combina con “todo lo que vuela come metal”, pero la etiqueta sigue siendo `si`. Se usa en la mezcla sintética de entrenamiento; la suite de evaluación por defecto no selecciona esa familia.

### 2.4 `synth_share = 0` todavía puede dejar mezcla sintética

`corpus.py` fija un mínimo de 2.000 bloques generados y selecciona herramientas con un mínimo solicitado de 3.000 cuando hay datos ([líneas 140–145](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/corpus.py#L140-L145)), aunque la configuración permita `synth_share` igual a cero ([líneas 132–133](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/corpus.py#L132-L133)). La opción no garantiza desactivar la sintética.

### 2.5 `eval.py --tasks ""` no valida que el conjunto de tareas esté vacío

La función `items(..., [])` produce un pool vacío y más abajo se divide por `len(pool)` ([`nexus/src/eval.py`, líneas 43–47 y 96–101](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/eval.py#L43-L47)). Con un motor configurado, ese caso llega a `ZeroDivisionError`; la suite ordinaria no lo cubre.

### 2.6 Verificación: guard numérico acotado más crítico LLM, no verificador general

La arquitectura describe correctamente que el guard sólo corrige operaciones extraíbles y no es un parser general ([`ARCHITECTURE.md`, línea 18](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ARCHITECTURE.md#L18)). Fuera de ese caso, el crítico acepta cualquier texto con `PASS` y sin `REVISA` ([`agent.py`, líneas 303–312](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L303-L312)). Reproduje un falso positivo con una frase que decía que el resultado era incorrecto pero contenía la palabra `PASS`.

**Lectura correcta de la descripción:** hay comprobaciones, pero no una garantía general de verdad. La arquitectura lo advierte; las etiquetas `verify_ok` y “verified” deben leerse con ese límite.

### 2.7 El reporte de `/api/status` no comunica si una tarea administrativa sigue activa

La API mantiene `_TASKS` para impedir dos procesos de entrenamiento/evaluación simultáneos ([`nexus/src/server.py`, líneas 23–24 y 303–327](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L23-L24)). Sin embargo, `status()` devuelve estado del motor, documentos, logs de entrenamiento y reporte de evaluación, pero no incluye `_TASKS` ([líneas 57–86](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/server.py#L57-L86)).

**Diferencia:** el endpoint muestra progreso/logs disponibles, pero no expone explícitamente `train_running` o `eval_running`; la UI no puede confirmar el estado real del subproceso sólo consultando ese objeto.

### 2.8 La hoja de ruta dice “registro completo” de trazas; el almacenamiento es parcial

La hoja de ruta promete un “Registro completo de trazas” en SQLite para auditoría ([`ROADMAP.md`, líneas 79–82](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ROADMAP.md#L79-L82)). El agente sólo persiste la traza de la respuesta elegida, no las trazas de todos los candidatos generados por autoconsistencia ([`agent.py`, líneas 322–357](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L322-L357)); además, `Memory.log_trace()` trunca el JSON de pasos a 4.000 caracteres ([`tools.py`, líneas 189–193](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/tools.py#L189-L193)).

**Resultado:** hay registro SQLite de la traza seleccionada, pero no es un registro completo de todas las generaciones ni garantiza conservar íntegros los pasos largos.

### 2.9 “Coincidencia exacta” del benchmark significa coincidencia tras normalización

La documentación llama a las puntuaciones “coincidencias exactas con los generadores” ([`ARCHITECTURE.md`, línea 36](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/docs/ARCHITECTURE.md#L36)). El benchmark compara `norm_ans(answer)` con `norm_ans(gold)` ([`eval.py`, línea 88](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/eval.py#L88)); esa función elimina espacios, ignora mayúsculas, retira punto final y normaliza ciertas cadenas numéricas ([`agent.py`, líneas 86–93](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/src/agent.py#L86-L93)).

**Lectura correcta:** es exact match normalizado, no igualdad literal carácter por carácter. La distinción es pequeña pero importa al interpretar “exactas”.

### 2.10 El repositorio incluye un tercer ejecutable NEXUS que los README no describen

La raíz incluye [`nexus_single.py`](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus_single.py), con modos `serve`, `ask` y `selftest` ([líneas 530–551](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus_single.py#L530-L551)), servidor HTTP propio, UI/SSE y SQLite propio. El README raíz presenta el repositorio como dos demostradores y explica NEXUS bajo `nexus/`, pero no documenta esta alternativa autocontenida. Las pruebas del repo sí la importan.

**Resultado:** no es un stub —tiene su propio modo `selftest`—, pero sí es una ruta de uso presente en el código y ausente de la guía principal. Hay que tratarla como otra implementación, no asumir que utiliza los mismos endpoints o configuración que `nexus/src/server.py`.

## 3. Funcionalidades nombradas en la hoja de ruta que no están implementadas

La documentación de NEXUS aclara que la hoja de ruta no equivale a funciones actuales ([`nexus/README.md`, línea 48](https://github.com/donovan-hue/red_neuronal_ai/blob/00d9c8b83c81a80b6a58da95e3e7e6722a451c7d/nexus/README.md#L48)). Confirmé además que los términos siguientes aparecen en la hoja de ruta, pero no como implementación en el resto del commit:

- **Entrenamiento a escala:** Common Crawl/filtrado/deduplicación MinHash, corpus de cientos de miles de millones de tokens, `torchrun` + FSDP/BF16, recuperación automática de fallos de nodo.
- **Post-entrenamiento:** SFT con trazas destiladas, RLVR, DPO/ORPO.
- **Evaluación externa continua:** MMLU-Pro, GPQA, AIME, BFCL, SWE-bench-lite y tau-bench.
- **Producto/inferencia:** vLLM/SGLang, batching continuo, caché de prefijos, decodificación especulativa, cuantización FP8/AWQ.
- **Orquestación/datos:** enrutado por coste, reranker de RAG, caché semántica, herramientas con allow-list por usuario y límite de gasto por sesión.
- **Seguridad aún pendiente para producto:** contenedor aislado para Python, evaluación sistemática de jailbreak/sesgo, límites de gasto y despliegue con HTTPS/infraestructura supervisada.

Esto es trabajo propuesto, no funcionalidades que el README de NEXUS presente como ya terminadas. La excepción a aclarar es la frase del roadmap que dice que el enrutado por coste ya está esbozado en `make_engine`: ese punto sí contradice el código actual (hallazgo 1.2).

## 4. Secciones cuya descripción sí coincide con el código

Para no marcar como errores lo que sí está bien documentado:

- El README raíz separa correctamente el demostrador clásico del NEXUS; la API clásica guarda/carga NPZ con `allow_pickle=False`.
- La API clásica valida estructura y rangos de entrada, limita el cuerpo y protege `/entrenar` con `API_TRAIN_TOKEN` y `X-API-Train-Token`.
- El README de NEXUS avisa que la ejecución de Python está deshabilitada y que el servicio no es para exposición pública sin controles adicionales.
- La arquitectura advierte expresamente que el SSE no es generación token a token.
- El benchmark crea memoria temporal y desactiva la recuperación de ejemplares entre ítems.
- Los resultados de `docs/RESULTS.md` se califican como históricos/no reproducidos por la revisión actual.
- La descripción BM25, SQLite, proveedor API por variables de entorno y límites del corpus literario corresponde en general a los componentes implementados.

## Alcance de la verificación

Este reporte compara el código fuente y los documentos del commit remoto citado. No se arrancó el runtime NEXUS completo ni se conectaron proveedores API: en el entorno faltaban `torch`, `tokenizers` y claves de proveedor. Las reproducciones de tokenizer, generador y verificador se señalan como tales; el resto de conclusiones proviene de rutas de código estáticas y está etiquetado con sus límites.

El archivo queda en el workspace local. No edité ni publiqué código del proyecto y no hice commit ni push.

# NEXUS

Prototipo de agente con un transformer local opcional, proveedores API, herramientas aritméticas, RAG BM25, memoria SQLite y benchmark reproducible. No es un modelo frontier ni un servicio listo para exposición pública.

## Inicio

```bash
cd nexus
./run.sh setup
./run.sh train
./run.sh serve
```

`setup` instala dependencias y descarga el corpus de Project Gutenberg cuando hace falta. El entrenamiento guarda checkpoints bajo `models/` y los datos bajo `data/`; ambos están excluidos de Git.

Para iniciar el servidor sin entrenar, descarga el checkpoint publicado:

```bash
./get_model.sh
./run.sh serve
```

El descargador verifica SHA-256 completo y crea el vocabulario y una partición de validación no solapada a partir del artefacto de entrenamiento publicado.

## Benchmark

```bash
./run.sh eval --arms rawlm,lm_cot,tools,tools+router --n 6
```

Brazos aceptados: `rawlm`, `lm_cot`, `tools`, `tools+router` (también se aceptan los alias `lm`, `cot` y `agent`). Los resultados se escriben en `runs/eval_report.json`. El benchmark crea memoria aislada y no reutiliza trayectorias entre ítems.

Los resultados históricos de `docs/RESULTS.md` corresponden a otra revisión y **no deben interpretarse como resultado reproducido por el código actual**.

## Proveedor API

Define la clave correspondiente (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` o `GEMINI_API_KEY`). Para habilitar llamadas desde el servidor configura también `NEXUS_API_TOKEN`; el cliente web solicitará ese token. Las llamadas a APIs pueden tener coste.

## Seguridad y límites

- Las operaciones de entrenamiento, evaluación y descarga de archivos de `runs/` requieren `NEXUS_ADMIN_TOKEN` en el encabezado `X-Nexus-Admin-Token`.
- `/api/config` es de solo lectura; los cambios se realizan fuera del servidor.
- El servicio limita tamaño y estructura de solicitudes.
- La herramienta de ejecución de Python está **deshabilitada**. `python -I` no constituye un sandbox.
- NEXUS no debe exponerse públicamente sin HTTPS, límites de tasa/gasto y controles de despliegue adicionales.
- El corpus por defecto es literatura, no una base factual actualizada.

La descripción de arquitectura está en `docs/ARCHITECTURE.md`; los pendientes de desarrollo están en [`../PENDIENTES.md`](../PENDIENTES.md).

Las trazas de errores del chat se registran en el servidor y no se envían al cliente. Protege el acceso a los logs, que pueden contener datos sensibles. Por defecto el límite de solicitudes es por proceso; se puede habilitar un limitador compartido local con `NEXUS_RATE_DB`.

El servidor limita el chat a 10 solicitudes por IP y minuto y a cuatro
conversaciones simultáneas por proceso; las operaciones administrativas comparten un límite de
10 solicitudes por IP y minuto. Los rechazos y acciones administrativas se registran
sin tokens ni contenido de solicitudes. Sin configuración adicional estos controles son por proceso, no limitan el gasto
y no reemplazan un proxy HTTPS o un control externo de presupuesto. Si se usa un proxy, configurar allí los límites por IP real;
el servidor solo observa la dirección del cliente conectado directamente.

### Memoria en el servidor web

El chat web usa memoria temporal aislada por solicitud y no guarda hechos ni
trayectorias entre peticiones. El historial enviado por el navegador mantiene el
contexto de la conversación. Esta medida evita compartir datos entre usuarios
mientras no exista autenticación individual. Las herramientas y programas de
línea de comandos conservan su configuración de memoria independiente.

El endpoint administrativo `GET /api/tasks` acepta `X-Nexus-Admin-Token` y
muestra el estado de la última tarea de entrenamiento y evaluación en el proceso
actual (`idle`, `running`, `succeeded`, `failed` o `timed_out`). Este estado es
volátil: se pierde al reiniciar y no se comparte entre procesos.

### Acceso individual opcional

Por defecto el servidor escucha solo en `127.0.0.1`. Para probar el acceso
individual, crea claves aleatorias en un directorio privado **fuera de Git**:

```sh
umask 077
python3 src/auth.py --file "$HOME/nexus-users.json" --user operador --role admin
python3 src/auth.py --file "$HOME/nexus-users.json" --user invitado --role user
export NEXUS_AUTH_FILE="$HOME/nexus-users.json"
./run.sh serve
```

Cada comando imprime una clave una sola vez; entrégala al usuario de forma
segura. El archivo guarda únicamente huellas SHA-256 de claves aleatorias, no
contraseñas. Con `NEXUS_AUTH_FILE` activado, todas las rutas `/api/` requieren
`Authorization: Bearer <clave>`; solo el rol `admin` accede a tareas y
archivos administrativos. La interfaz solicita la clave y la mantiene en
`sessionStorage`. El token separado `NEXUS_API_TOKEN` sigue siendo obligatorio
para proveedores de pago. Si el archivo es inválido o inaccesible, la API falla
cerrada. Las claves nuevas caducan en 90 días por defecto; `--ttl-days` permite cambiar
la vigencia entre 1 y 3650 días. Los cambios se leen en cada petición. Para rotar
una clave y revocar inmediatamente la anterior, o revocar el usuario:

```sh
python3 src/auth.py --file "$HOME/nexus-users.json" --user invitado --rotate
python3 src/auth.py --file "$HOME/nexus-users.json" --user invitado --revoke
```

Las claves creadas antes de incorporar caducidad no tienen fecha de expiración;
rótalas para establecerla. La rotación imprime el nuevo token una sola vez.

Esto no implementa inicio de sesión con contraseñas, rotación automática,
autenticación multifactor ni límites distribuidos. El servidor no
implementa TLS: fuera de localhost usa un proxy HTTPS confiable y controles de
red adicionales. El archivo es sensible incluso si solo contiene huellas; no lo
subas a Git ni ejecutes varias escrituras simultáneas con el comando de alta.

### HTTPS mediante proxy

Para una instalación propia, `deploy/Caddyfile` es una plantilla de proxy HTTPS.
Requiere un dominio válido apuntando al servidor, Caddy instalado y acceso a los
puertos 80 y 443 para emitir el certificado. Configura autenticación individual,
`NEXUS_API_TOKEN` para proveedores de pago y mantén NEXUS en `127.0.0.1:8080`.
Exporta `NEXUS_DOMAIN` e inicia Caddy con esa plantilla. La plantilla no añade
control de acceso de red, presupuestos ni límites distribuidos; no constituye una
certificación de seguridad para exposición pública. Protege el acceso al archivo
de usuarios y a los logs, que pueden contener datos sensibles.

Consulta [seguridad y retención de datos](docs/SECURITY.md) para conocer límites y responsabilidades operativas.

### Límites compartidos y cuota de llamadas

Para procesos en **el mismo host**, usa un archivo SQLite en un directorio
privado local con bloqueo fiable:

```sh
umask 077
mkdir -p "$HOME/.nexus-private"
chmod 700 "$HOME/.nexus-private"
export NEXUS_RATE_DB="$HOME/.nexus-private/limits.sqlite"
export NEXUS_API_DAILY_CALL_LIMIT=100
```

Todos los procesos que deban compartir las cuotas tienen que usar el mismo
archivo y la misma configuración. El límite diario es global entre proveedores,
se reinicia a medianoche UTC y reserva una llamada antes de cada solicitud,
incluso si luego falla. Si el almacenamiento falla o se configura una cuota sin
base de datos, se bloquean las llamadas. El límite HTTP falla cerrado si la
base no está disponible. Las consultas pueden bloquearse brevemente por la
contención SQLite. No uses este método como limitador entre hosts o sobre un
filesystem que no garantice bloqueos SQLite.

**La cuota cuenta llamadas, no dinero ni tokens**: no garantiza un límite de
gasto. Configura también los presupuestos y límites de facturación del proveedor.
El límite de conversaciones simultáneas sigue siendo por proceso. Si hay un
proxy inverso, los límites por IP observan la IP del proxy; aplica allí límites
por cliente con una configuración de confianza correcta. La base contiene
direcciones y contadores: protégela y gestiona su retención.

### Modo privado individual (incluido Termux)

`./run-private.sh` inicia el servidor en `127.0.0.1`, exige una clave individual,
utiliza el limitador SQLite local y fuerza el motor local. Aunque existan claves
de proveedores o la configuración indique una API, el motor de proveedores
rechaza sus llamadas con `NEXUS_PRIVATE_MODE=1`. No es un firewall: el sistema,
la instalación de dependencias, otras aplicaciones o funcionalidades ajenas
pueden usar internet. No evita el acceso de alguien que controle el teléfono.

Para preparar el acceso desde la carpeta `nexus/`:

```sh
umask 077
mkdir -p "$HOME/.nexus-private"
chmod 700 "$HOME/.nexus-private"
python3 src/auth.py --file "$HOME/.nexus-private/users.json" --user propietario --role admin
./run-private.sh
```

Guarda la clave que se muestra en un gestor seguro; no la compartas. Abre
`http://127.0.0.1:8080` en el navegador del mismo teléfono y proporciónala cuando
se solicite. El puerto puede cambiarse con `PORT`. La sesión del navegador guarda
la clave en `sessionStorage`, no en el repositorio; cierra la pestaña al terminar.
Para apagar el servidor usa Ctrl+C. La autenticación usa claves con vencimiento;
consulta la sección de acceso individual para rotarlas o revocarlas.

Para usar el modelo local se necesitan PyTorch, el tokenizador y un checkpoint
compatible. `./run-private.sh` llama a `./run.sh serve`, que puede intentar
instalar dependencias si faltan; la primera instalación y `./get_model.sh`
necesitan internet. **No se ha verificado la instalación de PyTorch ni el
rendimiento del modelo en Termux**: dependen del teléfono, Android y los paquetes
compatibles disponibles. Si falla la instalación no afirmamos que el modo local
funcione en ese dispositivo. El MCP de KRONOS es independiente y todavía no
está conectado a NEXUS.

Localhost limita la exposición de red, pero no aísla completamente otros
procesos del teléfono. Usa bloqueo de pantalla, protege las copias de seguridad,
no compartas credenciales y revisa los archivos locales y logs. Este modo no
cifra por sí mismo esos archivos. No uses un proxy público ni cambies el modo
privado para escuchar en todas las interfaces.

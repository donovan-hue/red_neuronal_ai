# Seguridad y datos

NEXUS es un prototipo y no debe exponerse públicamente sin una revisión del
entorno, HTTPS, controles de acceso y límites de gasto compartidos.

## Datos y retención

- El chat web usa memoria temporal por solicitud. El historial que envía el
  navegador permanece bajo control del cliente; la interfaz lo conserva en
  memoria durante la sesión de la página.
- El modo de línea de comandos puede guardar hechos y trayectorias en SQLite.
  Los archivos de modelos, memoria y logs están en `models/` y `runs/`.
- Las tareas administrativas producen logs en `runs/`. Las excepciones pueden
  escribirse en el log del proceso y contener datos sensibles. No hay borrado
  automático: el operador debe definir plazos y eliminar los archivos de forma
  segura conforme a su política. No registrar secretos deliberadamente.
- El archivo de usuarios contiene huellas de claves; protege sus permisos y no
  lo guardes en el repositorio. La copia de seguridad también debe protegerse.

## Límites conocidos

- Los límites de velocidad son por proceso por defecto; `NEXUS_RATE_DB` permite
  compartirlos entre procesos en un mismo host. La concurrencia es por proceso.
  Un proxy puede hacer que varios clientes compartan dirección.
- La cuota opcional cuenta llamadas, no dinero ni tokens. No hay presupuesto
  monetario ni limitador entre hosts; configura límites en el proveedor.
- Las respuestas RAG y las observaciones de herramientas se tratan como datos
  no confiables, pero eso no garantiza protección contra prompt injection.
- La ejecución de Python está deshabilitada. No debe habilitarse sin un sandbox
  independiente, con aislamiento de red, CPU, memoria, procesos y archivos.
- La plantilla Caddy no sustituye auditoría, gestión segura de certificados,
  control de acceso de red ni monitorización del despliegue.
- Las claves sin fecha de caducidad creadas por versiones anteriores deben
  rotarse. Una clave comprometida debe revocarse inmediatamente.

## Modo privado

`run-private.sh` exige autenticación, fuerza loopback y rechaza llamadas de los
motores API; no proporciona sandbox de procesos, cifrado de almacenamiento ni
bloqueo de tráfico saliente a nivel del sistema. Instalar dependencias o
actualizar artefactos puede requerir red. Proteger la sesión y el dispositivo
sigue siendo responsabilidad del propietario.

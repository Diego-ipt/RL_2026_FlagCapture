# RL_2026_FlagCapture

Capture the Flag multiagente en grilla 2D (PettingZoo Parallel API) para entrenar equipos con aprendizaje por refuerzo (MAPPO).

## Reglas del juego

- Dos equipos (azul a la izquierda, rojo a la derecha), cada uno con base y bandera.
- **Mapa:** cajas y murallas simétricas (espejo), nuevo en cada ronda. Siempre conexo y con las bases despejadas.
- **Ocupación:** nadie entra a una celda ocupada por otro agente, salvo las bases y las celdas con una bandera en el suelo.
- **Camino garantizado:** para cada equipo siempre existe un camino de movimientos válidos desde la bandera rival hasta su base.
  - El camino se guarda en caché. Solo se recalcula con BFS cuando un agente lo pisa: el movimiento es válido si queda otro camino, y si no se rechaza.
  - No se puede tapar un pasillo único ni rodear una base.
  - Si tras un evento de bandera no hay camino, quien lo tapa queda obligado a moverse. Elige a dónde, pero solo entre movimientos válidos.
- **Acciones:** 0 quieto, 1–4 mover (N/S/E/O), 5 **quitar**, 6 **pasar**.
- **Recoger:** pisar la bandera rival (en su base o en el suelo) la recoge, en cualquier momento.
- **Anotar:** llevar la bandera rival a tu base gana la ronda. No hace falta que tu bandera esté en casa. **Anotar se resuelve antes que quitar.**
- **Quitar:** solo a un portador rival adyacente que esté **en tu mitad**. La bandera cae al suelo y el portador no puede recoger durante 2 pasos.
- **Bandera en el suelo:** si la toca un agente de su equipo, vuelve a su base (niega el avance). Si la toca un rival, la recoge. Si la tocan ambos a la vez, vuelve.
- **Pasar:** el portador entrega la bandera a un aliado adyacente.
- La ronda termina al anotar o al cumplirse `max_steps`.
- Cada paso se resuelve en este orden: mover → anotar → quitar → pasar → contactos con banderas.
- **Máscara de acciones:** `infos[a]["action_mask"]` marca qué acciones son válidas.

## Recompensa

El equipo azul recibe en cada paso:

```
S_e(s) = w_def·(1 − d(bandera_e, base_e)/D) + w_atk·(1 − d(bandera_rival, base_e)/D)
M(s)   = β·(S_azul(s) − S_rojo(s))         marcador de posición de banderas
M(s_final) = 0                              al terminar la ronda el marcador no cuenta

r_azul(t) = M(s_{t+1}) − M(s_t) + R_term    R_term = +r_win si anota azul, −r_win si anota rojo
r_rojo(t) = −r_azul(t)                      suma cero exacta
```

- d es la distancia de camino real (BFS respetando paredes), normalizada por la máxima del mapa.
- El marcador sube cuando tu bandera está cerca de tu base y cuando la bandera enemiga se acerca a tu base. Se paga solo su **cambio**, no se acumula por paso.
- Como el marcador vuelve a 0 al final, el retorno de una ronda es exactamente **±r_win o 0**. Retener la bandera no da puntos.
- **β pierde importancia con el entrenamiento:** `β = beta0·(1 − epoch/anneal_epochs)` hasta 0. Opción: `decay: episode` lo baja dentro de cada ronda.

## Agente (MAPPO, CTDE)

- **Actor** compartido por todo el equipo (parameter sharing), MLP 128-128.
  - Entrada, observación local:
    - ventana (2r+1)² con 7 canales: paredes, aliados, rivales, bandera propia, bandera rival, aliado portador y rival portador;
    - ID one-hot del agente (permite roles distintos);
    - si porta, tiempo, posición propia, cooldown y si está obligado a moverse;
    - estado (base/llevada/suelo) y posición de ambas banderas.
  - Salida: 7 acciones; las inválidas se enmascaran (logit −∞).
- **Crítico centralizado:** recibe el estado global (`env.state()`) + el ID del agente y estima V. Solo se usa durante el entrenamiento.
- **Entrenamiento:**
  - GAE (λ=0.95);
  - PPO con clip 0.2 y value clipping;
  - normalización de valores y entropía 0.01;
  - checkpoints y métricas por época en `runs/<run>/`.
- **Vista egocéntrica:** el rojo ve el mapa espejado (su base a la izquierda) y sus acciones E/O se intercambian, así que ambas redes reciben el mismo formato de observación.
- **Dos MAPPO distintos** (`opponent: selfplay`): el azul y el rojo tienen cada uno su actor, crítico y optimizador; no comparten pesos.
  - En cada partida juegan los dos aprendices actuales y ambos aprenden.
  - Con prob. 0.3, uno de ellos enfrenta en cambio a una versión pasada del otro equipo, elegida por PFSP (más probable cuanto peor le va contra ella).

## Estructura

- `src/ctf/env.py`: entorno `CTFParallelEnv`.
- `src/ctf/maps.py`, `src/ctf/paths.py`: generador de mapas y BFS.
- `src/ctf/checkpoint.py`: `RunStore` (checkpoints, `metrics.csv`, `index.json`).
- `src/ctf/mappo/`: actor, crítico, buffer/GAE, políticas rivales y trainer.
- `configs/`: configuraciones de entrenamiento.
- `scripts/`: `train.py`, `evaluate.py`, `watch.py` (GIF o ventana, con HUD del marcador), `marker_trace.py`, `plot.py`.
- `tests/`: tests con pytest (API, reglas, suma cero, trainer).

## Uso

```
docker compose build
docker compose run --rm ctf pytest -q tests
docker compose run --rm ctf python scripts/train.py --config configs/mappo_v2_rules.yaml
docker compose run --rm ctf python scripts/train.py --run-name v2_2v2 --set env.n_per_team=2 env.size=10
docker compose run --rm ctf python scripts/train.py --config configs/selfplay_2v2_walls.yaml
docker compose run --rm ctf python scripts/evaluate.py --a sp2_2v2_walls --b sp2_2v2_walls:300   # A vs B ('random', RUN o RUN:EPOCH)
docker compose run --rm ctf python scripts/marker_trace.py --blue sp2_2v2_walls --red sp2_2v2_walls:300
docker compose run --rm ctf python scripts/watch.py --blue sp2_2v2_walls --red sp2_2v2_walls --stochastic --out runs/viz/sp2.gif
docker compose run --rm ctf python scripts/plot.py --runs sp2_2v2_walls --keys win_blue win_red draw_rate strips_blue strips_red
```

Para ver una ventana en vivo, corre `scripts/watch.py --human` fuera de Docker (requiere `pygame-ce`, `gymnasium`, `pettingzoo`, `numpy`, `torch` y `PYTHONPATH=src`).

## Apéndice: decisiones de diseño

Decisiones tomadas durante el desarrollo, con su motivo. Las marcadas **(equipo)** fueron definidas por los autores del proyecto; las marcadas **(técnica)** se tomaron al implementar para que esas reglas funcionaran.

### Reglas del juego

| Decisión                                                                                                                                                                                            | Motivo                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(equipo)** Se puede recoger la bandera rival en cualquier momento: en su base o en el suelo.                                                                                                 | Juego más dinámico que el CTF clásico de "solo en la base".                                                                                                                                             |
| **(equipo)** Acción **quitar**: la bandera del portador cae al suelo.                                                                                                                   | Permite defender sin eliminar agentes.                                                                                                                                                                     |
| **(equipo)** Bandera en el suelo: si la toca su equipo vuelve a su base; si la toca el rival, la recoge.                                                                                       | Recuperar la bandera niega el avance del rival.                                                                                                                                                            |
| **(equipo)** Acción **pasar** la bandera a un aliado adyacente.                                                                                                                         | Permite cooperación ofensiva.                                                                                                                                                                             |
| **(equipo)** Para anotar no hace falta tener la bandera propia en casa.                                                                                                                        | Evita bloqueos mutuos en que ningún equipo puede anotar.                                                                                                                                                  |
| **(equipo)** **Anotar se resuelve antes que quitar.**                                                                                                                                    | En el primer self-play, un defensor acampaba sobre la base y quitaba la bandera justo cuando el portador llegaba. El 70–95% de las partidas entre versiones entrenadas terminaba en empate (punto silla). |
| **(equipo)** **Quitar solo en tu propia mitad** (CTF clásico).                                                                                                                          | Al cruzar a su mitad, el portador queda a salvo. La defensa ocurre en territorio propio.                                                                                                                   |
| **(equipo)** **Siempre existe un camino de movimientos válidos desde la bandera rival hasta la base de cada equipo.** Nadie puede pararse justo en la base ni rodearla para bloquearla. | Elimina el bloqueo físico de la base.                                                                                                                                                                     |
| **(equipo)** El camino se guarda en caché y **solo se recalcula si un agente lo pisa**: el movimiento vale si existe otro camino, y si no se rechaza.                                   | Un BFS en cada paso escala mal con mapas grandes o difíciles.                                                                                                                                             |
| **(equipo)** Si tras un evento de bandera no hay camino, el agente que lo tapa **debe moverse sí o sí**, pero elige a dónde, siempre que sea un movimiento válido.                   | Mantener la garantía sin quitarle la decisión al agente. Se implementó con máscara de acciones.                                                                                                        |
| **(equipo)** Las **bases se comparten** (varios agentes pueden estar en ellas).                                                                                                          | Si no, quien entra a robar en la base rival taparía el final del camino del rival.                                                                                                                        |
| **(técnica)** Las celdas con una bandera en el suelo también se comparten.                                                                                                                   | Si no, al portador que pierde la bandera la tiene debajo y el dueño nunca podría pisarla para devolverla.                                                                                                |
| **(técnica)** El portador al que le quitan la bandera no puede recogerla durante 2 pasos (cooldown).                                                                                          | Si no, la recogería de nuevo en el mismo paso y quitar no tendría efecto.                                                                                                                                |
| **(equipo)** Mapa con **bloqueos**: cajas y murallas.                                                                                                                                    | Juego más táctico.                                                                                                                                                                                       |
| **(técnica)** Mapas simétricos en espejo, conexos y con las bases despejadas; uno nuevo en cada ronda.                                                                                       | Ningún equipo tiene ventaja de mapa, siempre hay camino entre cualquier par de celdas libres, y las políticas generalizan en vez de memorizar un mapa.                                                   |

### Recompensa

| Decisión                                                                                                                                                     | Motivo                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(equipo)** Hay un puntaje según qué tan cerca está **tu bandera de tu base** y qué tan cerca está **la bandera enemiga de tu base**. | Da una señal densa ligada al objetivo del juego.                                                                                                                                             |
| **(equipo)** **Suma cero:** lo que suma un equipo se le resta al otro.                                                                            | Juego competitivo puro; evita la colusión entre equipos.                                                                                                                                     |
| **(equipo)** El puntaje de posición es un **marcador** que se mantiene según dónde estén las banderas, **no se acumula por paso**.      | Retener la bandera no debe dar puntos (evita el "flag camping"). Por eso se paga solo el cambio del marcador.                                                                                 |
| **(equipo)** **Al terminar la ronda el marcador no cuenta:** solo cuenta la victoria.                                                             | El retorno de una ronda es exactamente ±r_win o 0. Es la forma de reward shaping basada en potenciales (Ng et al., 1999), que no cambia la política óptima.                                |
| **(equipo)** El marcador **pierde importancia con el tiempo**.                                                                                    | Al inicio guía la exploración y al final el agente optimiza solo la victoria. β baja a 0 a lo largo del entrenamiento (`anneal_epochs`); opcionalmente puede bajar dentro de cada ronda. |
| **(equipo)** **Sin incentivos de rol individuales** (por ejemplo, "premiar al defensor por quedarse en base").                                    | Son heurísticas que invalidan cualquier afirmación sobre comportamiento emergente. Los roles deben surgir del ID one-hot + el crítico centralizado + el self-play.                         |
| **(técnica)** La distancia del marcador es la de camino real (BFS con paredes), no Manhattan.                                                          | Con murallas, la distancia Manhattan engaña sobre qué tan cerca está algo.                                                                                                                 |
| **(técnica)** El límite de tiempo se trata como final real de la ronda (sin bootstrap del valor).                                                     | La ronda realmente termina, vale 0, y el tiempo restante va en la observación.                                                                                                               |

### Agentes y entrenamiento

| Decisión                                                                                                              | Motivo                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **(equipo)** MAPPO implementado desde cero, con CTDE: actor con observación local y crítico con estado global. | Base del proyecto. MAPPO es estable en espacios discretos (Yu et al., 2022).                                    |
| **(equipo)** **Dos MAPPO distintos**, uno para el azul y otro para el rojo, sin compartir red.             | Evaluar dos aprendices independientes que compiten.                                                             |
| **(técnica)** Dentro de cada equipo, actor compartido + ID one-hot por agente.                                  | Más datos por actualización, y el ID permite que surjan roles distintos.                                      |
| **(técnica)** Vista egocéntrica: el rojo ve el mapa espejado.                                                  | Ambos aprendices enfrentan exactamente el mismo problema. Se verificó con 18,000 estados espejados idénticos. |
| **(técnica)** Liga por equipo con PFSP: con prob. 0.3 un aprendiz enfrenta una versión pasada del rival.       | Evita el olvido de estrategias y los ciclos del self-play puro.                                                 |
| **(técnica)** Máscara de acciones (logits −∞ para las inválidas).                                           | Aplica la regla de "moverse sí o sí" sin quitar la decisión a la red, y evita desperdiciar acciones.         |
| **(equipo)** Todo corre en Docker; los pesos se guardan por época en `runs/<run>/` para comparar instancias.  | Reproducibilidad, y poder ver cómo escala el entrenamiento entre épocas.                                      |

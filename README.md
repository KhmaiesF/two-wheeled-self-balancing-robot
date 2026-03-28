# 🤖 Robot Auto-Équilibrant

Robot à deux roues auto-équilibrant simulé avec PyBullet et contrôlé par PID + PPO.

## 📁 Structure du Projet

```
robot_auto_equilibrant/
├── assets/
│   └── robot.urdf          # Modèle 3D du robot
├── src/
│   ├── environment.py      # Environnement Gymnasium
│   ├── pid_controller.py   # Contrôleur PID
│   ├── controller.py       # Hybrid controller (PID + PPO)
│   ├── train.py           # Entraînement PPO
│   ├── evaluate.py        # Évaluation multi-scénarios
│   ├── visualize.py       # Visualisation temps réel
│   ├── test.py            # Tests automatiques
│   └── convert_to_onnx.py # Export pour Raspberry Pi
├── models/                 # Modèles entraînés
├── real/                   # Code Raspberry Pi
├── logs/                   # Logs d'entraînement
└── README.md
```

## 🚀 Utilisation Rapide

### Installation

```bash
python -m venv venv
.\venv\Scripts\Activate  # Windows
pip install pybullet gymnasium stable-baselines3 numpy
```

### Visualisation (PID seul)

```bash
python -m src.visualize --mode pid --duration 30
```

### Tests

```bash
python -m src.test
```

### Entraînement PPO

```bash
python -m src.train --steps 300000
```

### Évaluation

```bash
python -m src.evaluate --mode pid+ppo
```

## 🎮 Modes de Contrôle

| Mode | Description |
|------|-------------|
| `pid` | PID seul (baseline) |
| `ppo` | PPO seul (from scratch) |
| `pid+ppo` | PID + correction PPO (recommandé) |

## 📊 Caractéristiques Physiques

- **Masse totale**: ~1.1 kg
- **Hauteur**: ~18 cm
- **Roues**: Ø75mm, largeur 20mm
- **Entraxe**: 270mm
- **Moteurs**: 2 Nm max

## 🔧 Paramètres Clés

### PID
- Kp = 25.0
- Ki = 0.0  
- Kd = 0.8

### Physique
- Friction latérale: 5.0 (empêche glissement)
- Friction de roulement: 0.001

## 📈 Performances

| Métrique | Valeur |
|----------|--------|
| Pitch max | < 0.01° |
| Vitesse roue max | ~12 rad/s |
| Survie (10s) | 100% |
| Récupération push | ✅ |

## 🎯 Objectifs du Robot

1. **Rester vertical** : angle = 0° en permanence
2. **Rester en place** : pas de dérive
3. **Répondre aux perturbations** : récupération après push
4. **Commandes lisses** : pas de vibrations des moteurs

## 📝 License

MIT

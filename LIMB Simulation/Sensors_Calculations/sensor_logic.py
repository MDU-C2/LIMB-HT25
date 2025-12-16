# sensor_logic.py (Version Optimisée Replay)
import math

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0 
        
    # MODIFICATION : On retire self.last_time et on demande dt en argument
    def update(self, accel_xyz, gyro_xyz, dt):
        ax, ay, az = accel_xyz
        gx, gy, gz = gyro_xyz # Assure-toi que c'est en RAD/S

        # 1. Angles Accéléro
        # Sécurité : éviter division par zéro
        norm_yz = math.sqrt(ay**2 + az**2)
        accel_pitch = math.degrees(math.atan2(ax, norm_yz)) if norm_yz > 0 else 0
        
        norm_xz = math.sqrt(ax**2 + az**2)
        accel_roll  = math.degrees(math.atan2(ay, norm_xz)) if norm_xz > 0 else 0
        
        # 2. Intégration Gyro
        self.pitch = self.alpha * (self.pitch + math.degrees(gy) * dt) + (1 - self.alpha) * accel_pitch
        self.roll = self.alpha * (self.roll + math.degrees(gx) * dt) + (1 - self.alpha) * accel_roll
        self.yaw = self.yaw + math.degrees(gz) * dt # Drift inévitable ici sans magnétomètre

        return {'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw}

class ArmController:
    def __init__(self):
        self.filter_arm = ComplementaryFilter()
        self.filter_forearm = ComplementaryFilter()
        self.is_calibrated = False
        self.offsets = {'arm': {'roll':0,'pitch':0,'yaw':0}, 'forearm': {'roll':0,'pitch':0,'yaw':0}}
        
        # SAFETY LIMITS (Degrés) - Pour ne pas casser PyBullet
        self.LIMITS = {
            "shoulder_x": (-90, 90),
            "shoulder_y": (-180, 180),
            "shoulder_z": (-90, 90),
            "elbow_x":    (-140, 0) # Le coude ne plie que dans un sens
        }

    def process(self, raw_data, dt=0.01):
        """
        raw_data : dict avec 'accel_1', 'gyro_1', etc.
        dt : delta time entre deux mesures (ex: 1/fréquence_imput)
        """
        a1, g1 = raw_data.get('accel_1'), raw_data.get('gyro_1')
        a2, g2 = raw_data.get('accel_2'), raw_data.get('gyro_2')
        
        # Update avec le dt explicite
        arm = self.filter_arm.update(a1, g1, dt)
        forearm = self.filter_forearm.update(a2, g2, dt)
        
        if not self.is_calibrated:
            self.offsets['arm'] = arm.copy()
            self.offsets['forearm'] = forearm.copy()
            self.is_calibrated = True
            return {k: 0.0 for k in ["shoulder_x", "shoulder_y", "shoulder_z", "elbow_x", "elbow_y"]}

        # Calcul des Deltas (inchangé mais propre)
        instructions = {
            "shoulder_x": arm['roll'] - self.offsets['arm']['roll'],
            "shoulder_y": arm['pitch'] - self.offsets['arm']['pitch'],
            "shoulder_z": arm['yaw'] - self.offsets['arm']['yaw'],
            # Pour le coude : Différence de pitch entre avant-bras et bras
            "elbow_x": -abs((forearm['pitch'] - self.offsets['forearm']['pitch']) - (arm['pitch'] - self.offsets['arm']['pitch'])), 
            "elbow_y": (forearm['roll'] - self.offsets['forearm']['roll']) - (arm['roll'] - self.offsets['arm']['roll']),
            "wrist_x": 0.0, "wrist_z": 0.0
        }
        
        # CLAMPING DE SÉCURITÉ (Indispensable pour la simu)
        for joint, value in instructions.items():
            if joint in self.LIMITS:
                m, M = self.LIMITS[joint]
                instructions[joint] = max(m, min(M, value))
                
        return instructions
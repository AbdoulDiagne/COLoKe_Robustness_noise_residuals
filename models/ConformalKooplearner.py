import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.linalg as la
import torch
from scipy.stats import norm
import random
from scipy.stats import chi2
import torch
class CKL(nn.Module):
    def __init__(self, train_args, CP_args):
        super().__init__()
        self.device = train_args["device"]    
        self.d = train_args["d"]
        self.L = train_args["L"]
        self.K = nn.Linear(self.d+self.L, self.d+self.L, bias=False)  # Learnable K matrix
        
        # model architecture
        if "architecture" in train_args:
            
            layers = []
            in_dim = self.d

            for hidden_dim in train_args["architecture"]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            
            # Final layer (to self.L)
            layers.append(nn.Linear(in_dim, self.L))
            
            self.encoder = nn.Sequential(*layers)



        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.d, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, self.L),
            )


        # training
        self.off_epochs = train_args["off_epochs"]
        self.max_iter = train_args["max_iter"]
        self.n_max = train_args["n_max"]
        self.lr = train_args["lr"]
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), lr = self.lr)
        self.X_t_list = []
        self.X_t_next_list = []
        self.losses = []
        self.iter_t = []
        self.test_error_evolution = []
        self.tracked_traj = None
        self.tracked_positions = []
        self.nb_outliers_total = 0
        self.total_timesteps = 0
        # conformal PI
        self.scores = []
        self.scores_after = []
        self.qs = []
        self.gs = []
        self.alpha = CP_args["alpha"]
        self.eta = CP_args["eta"]
        self.Csat = CP_args["Csat"]
        self.T_warm_up = CP_args["T_warm_up"]
        self.B_hat_horizon = self.T_warm_up-self.n_max
           
        # spectral properties
        self.eigenvalues = None
    #### encoder and forward prediction in latent space ####
    def encode_z(self, x):
        '''
        encode with projection
        '''
        return torch.cat((x, self.encoder(x)), dim=1)
    
    def forward(self, x, n_steps):
        '''
        encode to latent space and predict n steps
        '''
        z = self.encode_z(x)
        K_power_n = torch.matrix_power(self.K.weight, n_steps)
        z_n_pred = z @ K_power_n.T

        return z_n_pred
    
    def offline_data(self, X):
        '''
        X is a array of shape (N,T,D): N trajectories, T time steps and D stands for dimension
        '''
        for l in range(1, self.n_max+1):
            self.X_t_list.append(X[:, :-l, :].reshape(-1, self.d))        # x_{t-l}
            self.X_t_next_list.append(X[:, l:, :].reshape(-1, self.d))    # x_t
        # Convert to PyTorch tensors
        self.X_t_list = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in self.X_t_list]
        self.X_t_next_list = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in self.X_t_next_list]

    def online_data(self, X):
        '''
        X is a array of shape (N,T,D): N trajectories, T time steps and D stands for dimension
        '''
        self.X_t_list=[]
        self.X_t_next_list=[]
        for l in range(1, self.n_max + 1):
            self.X_t_list.append(X[:, :-l, :].reshape(-1, self.d))  # x_{t-l}
            self.X_t_next_list.append(X[:, l:, :].reshape(-1, self.d))  # x_t
        # Convert to PyTorch tensors
        self.X_t_list = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in self.X_t_list]
        self.X_t_next_list = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in self.X_t_next_list]

    def _offline_training(self):
        '''
        one iteration
        '''
        self.optimizer.zero_grad()
        loss = 0
        for l in range(1, self.n_max+1):
            z_pred = self.forward(self.X_t_list[l-1], l)

            z_true = self.encode_z(self.X_t_next_list[l-1])

            loss += self.criterion(z_pred, z_true)

            
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        
        
    def offline_training(self, epochs, trajs_test=None):
        '''
        training on offline data
        '''
        for epoch in range(epochs):
            self._offline_training()
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {self.losses[-1]}")


    def online_training(self, X_tilde, t):
        X_tilde = torch.tensor(X_tilde, dtype=torch.float32).to(self.device)
        s_t = self.score_function_chi2_residuals(X_tilde)

        print(f"score  {s_t}")
        self.scores.append(s_t)
        q_t = self.conformalPI(t) # update q_t+1 and return q_t
        print(f"seuil {q_t}")
        it = 0
        self.tracked_positions=[]
        s_t_true = s_t
        if s_t_true >= q_t:
            self.online_data(X_tilde)
            while s_t >= q_t and it <= self.max_iter:

                self._offline_training_chi2_residuals(X_tilde)

                s_t=self.score_function_chi2_residuals(X_tilde)

                it += 1
        else:
            for _ in range(1):
                self.online_data(X_tilde)
            it = 1
        self.iter_t.append(it)
        self.scores_after.append(s_t)
        if len(self.tracked_positions) == 0:
            return  self.tracked_traj
        return self.tracked_positions[0]
    def get_eigenvalues(self, delta_t):
        K = self.K.weight.detach().cpu().numpy()
        eigenvalues, _ = la.eig(K.T)
        self.eigenvalues = np.log(eigenvalues) / delta_t
        
        return self.eigenvalues
        
    def eigen_functions(self, x):
        z = self.encode_z(x)       
        K = self.K.weight.detach().cpu().numpy()
        _, eigenvectors = la.eig(K.T)
        P = eigenvectors      
        evalu = P.T @ z.detach().cpu().numpy().T
        return evalu
    
    def l_steps_error(self, X_tilde, l_step = 1):
        x_t_true = X_tilde[:, -1, :]
        x_t_l = X_tilde[:, -1 - l_step, :]
        z_t_pred = self.forward(x_t_l, l_step)    
        x_t_pred = z_t_pred[:,:self.d]    
        error = self.criterion(x_t_true, x_t_pred)
        return error.item()
        
    #### score function and Conformal PI procedure ####
    def score_function(self, X_tilde):
        score = 0.0
        
        x_t = X_tilde[:, -1, :]
        z_t_true = self.encode_z(x_t)
        
        for l in range(1, self.n_max + 1):
            x_t_l = X_tilde[:, self.n_max - l, :]
            z_t_pred = self.forward(x_t_l, l)    
            score += self.criterion(z_t_pred, z_t_true)
        
        return score.item()
    
    #### conformal PI 
    def mytan(self, x):
        if x >= np.pi/2:
            return np.infty
        elif x <= -np.pi/2:
            return -np.infty
        else:
            return np.tan(x)
    
    def saturation_fn_log(self, x, t, KI):
        if KI == 0:
            return 0
        tan_out = self.mytan(x * np.log(t+1)/(self.Csat * (t+1)))
        out = KI * tan_out
        return  out
        
    def conformalPI(self, t):
        s_t = self.scores[-1]
        q_t = self.qs[-1]
        err_t = 1 if s_t > q_t else 0
        B_hat = np.max(self.scores[-self.n_max-1:-1]) 
             
        x = np.sum(self.gs)
        g_t = err_t - self.alpha 
        self.gs.append(g_t)
        
        q_t_next = q_t + self.eta * B_hat * g_t + self.saturation_fn_log(x, t, B_hat)
        if q_t_next <= 0:
            q_t_next = q_t
        self.qs.append(q_t_next)
        
        return q_t
    
    def initialization(self, X_warm_up):
        X_warm_up = torch.tensor(X_warm_up, dtype=torch.float32).to(self.device)
        for t in range(self.n_max+1, self.T_warm_up):
            X_tilde = X_warm_up[:, t-self.n_max : t+1, :]
            self.scores.append(self.score_function(X_tilde))
        self.qs.append(np.quantile(self.scores, 1-self.alpha))
        
    #### evaluation 
    def compute_error(self, trajs):
        X_test = torch.tensor(trajs[:,:-1,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        Y_test = torch.tensor(trajs[:,1:,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        Y_pred = self.forward(X_test, 1)[:,:self.d]
        mse = ((Y_pred - Y_test) ** 2).mean().item()
        return mse



    def _compute_residual_chi2_loss(self, X_tilde, proportion_saine=0.8):
        """
        Centre-réduit les résidus par dimension, applique un filtre Chi² à
        chaque point (df = self.d). Si au moins une composante est abérante,
        le point complet est considéré comme non sain.
        """
        if not isinstance(X_tilde, torch.Tensor):
            X_tilde = torch.tensor(
                X_tilde, dtype=torch.float32, device=self.device
            )

        N_trajs, L_window, _ = X_tilde.shape
        N_preds = L_window - 1
        d = self.d  # d = 2 dans ce cas

        # ==========================================================
        # 1. Calcul des résidus
        # ==========================================================
        residus = torch.zeros((N_trajs, N_preds, d), device=self.device)

        for i in range(N_preds):
            x_past = X_tilde[:, i, :]
            z_pred = self.forward(x_past, n_steps=1)
            x_true = X_tilde[:, i + 1, :]
            residus[:, i, :] = x_true - z_pred[:, :d]

        # ==========================================================
        # 2. Moyenne et écart-type par dimension
        # ==========================================================
        moyenne = torch.mean(residus, dim=(0, 1))
        variance = torch.var(residus, dim=(0, 1), unbiased=False)
        ecart_type = torch.sqrt(torch.clamp(variance, min=1e-8))

        # ==========================================================
        # 3. Centrage-réduction des résidus
        # ==========================================================
        residus_standardises = (residus - moyenne) / ecart_type

        # ==========================================================
        # 4. Statistique Chi² PAR POINT (Somme sur la dimension d)
        #    La somme de d normales au carré suit un Chi²(d)
        # ==========================================================
        # Shape : (N_trajs, N_preds)
        chi2_points = torch.sum(residus_standardises ** 2, dim=2)

        # ==========================================================
        # 5. Seuil Chi² avec df
        # ==========================================================
        seuil = chi2.ppf(proportion_saine, df=d)
        seuil = torch.tensor(seuil, dtype=torch.float32, device=self.device)

        # ==========================================================
        # 6. Masque au niveau du POINT (True si le point est sain)
        #    Shape : (N_trajs, N_preds) -> qu'on étend à (N_trajs, N_preds, d)
        # ==========================================================
        mask_points = chi2_points <= seuil  # (N_trajs, N_preds)
        mask_residus = mask_points.unsqueeze(-1).expand(
            -1, -1, d
        )  # (N_trajs, N_preds, d)

        # ==========================================================
        # 7. Loss de chaque trajectoire (avec les points uniquement sains)
        # ==========================================================
        residus_carres = residus ** 2
        residus_carres_sains = torch.where(
            mask_residus, residus_carres, torch.zeros_like(residus_carres)
        )

        # Compter le nombre de points sains (1 point = d résidus)
        nombre_points_sains = mask_points.sum(dim=1)  # (N_trajs,)
        nombre_residus_sains = nombre_points_sains * d

        nombre_residus_sains_safe = torch.clamp(nombre_residus_sains, min=1)

        # Loss = moyenne des résidus au carré pour les points sains de la trajectoire
        losses = (
                torch.sum(residus_carres_sains, dim=(1, 2)) / nombre_residus_sains_safe
        )

        # ==========================================================
        # 8. Trajectoires avec au moins UN point sain
        # ==========================================================
        mask_trajectoires = nombre_points_sains > 0

        return losses, mask_residus, mask_trajectoires, chi2_points, seuil,residus

    def _offline_training_chi2_residuals(
            self, X_tilde, proportion_saine=0.8
    ):
        """
        Met à jour les poids uniquement à partir des points sains (points dont
        toutes les coordonnées sont sous le seuil).
        """
        self.optimizer.zero_grad()

        # 1. Masquage SANS gradient
        with torch.no_grad():
            _, mask_residus, mask_trajectoires, _, _,_ = (
                self._compute_residual_chi2_loss(X_tilde, proportion_saine)
            )

        # 2. Recalcul différentiable des résidus
        N_trajs, L_window, _ = X_tilde.shape
        N_preds = L_window - 1

        residus = torch.zeros((N_trajs, N_preds, self.d), device=self.device)

        for i in range(N_preds):
            x_past = X_tilde[:, i, :]
            z_pred = self.forward(x_past, n_steps=1)
            x_true = X_tilde[:, i + 1, :]
            residus[:, i, :] = x_true - z_pred[:, : self.d]

        # 3. Application du masque aux points complets
        residus_carres = residus ** 2
        residus_carres_sains = torch.where(
            mask_residus, residus_carres, torch.zeros_like(residus_carres)
        )

        # 4. Calcul de la loss normalisée par le nombre de composantes saines
        nombre_residus_sains = mask_residus.sum(dim=(1, 2))
        losses = torch.sum(residus_carres_sains, dim=(1, 2)) / torch.clamp(
            nombre_residus_sains, min=1
        )

        # 5. Calcul de la loss finale (Moyenne si trajectoires saines, sinon Médiane)
        losses_saines = losses[mask_trajectoires]

        if losses_saines.numel() == 0:
            losses_brutes = torch.mean(residus ** 2, dim=(1, 2))
            loss_reconstruction = torch.median(losses_brutes)

        else:
            loss_reconstruction = torch.mean(losses_saines)

        # 6. Rétropropagation
        loss_reconstruction.backward()
        self.optimizer.step()

        self.losses.append(loss_reconstruction.item())
        return loss_reconstruction.item()

    def score_function_chi2_residuals(self, X_tilde, proportion_saine=0.8):
        """
        Calcule le score final sur l'ensemble du dataset en filtrant les points abérrants.
        """
        with torch.no_grad():
            losses, _, mask_trajectoires, _, _,residus_brutes = self._compute_residual_chi2_loss(
                X_tilde, proportion_saine
            )


            filtered_losses = losses[mask_trajectoires]

            if filtered_losses.numel() == 0:
                losses_brutes = torch.mean(residus_brutes ** 2, dim=(1, 2))
                final_score = torch.median(losses_brutes).item()
            else:
                final_score = torch.mean(filtered_losses).item()

        return final_score
    """def score_function_per_trajectory_ignore(self, X_tilde):
    
        Calcule un score global robuste en ignorant les trajectoires aberrantes.
        X_tilde: (N_trajs, window, D)
        return: float (score moyen ou médian filtré)
        

        if not isinstance(X_tilde, torch.Tensor):
            X_tilde = torch.tensor(X_tilde, dtype=torch.float32).to(self.device)

        N_trajs = X_tilde.shape[0]
        individual_scores = torch.zeros(N_trajs, device=self.device)

        with torch.no_grad():
            # 1. Encodage du point cible (réel)
            z_t_true = self.encode_z(X_tilde[:, -1, :])

            # 2. Calcul du score cumulé par trajectoire sur n_max horizons
            for l in range(1, self.n_max + 1):
                x_input = X_tilde[:, -1 - l, :]
                z_pred = self.forward(x_input, l)

                # Distance par trajectoire : forme (N_trajs,)
                dist = torch.mean((z_pred - z_t_true) ** 2, dim=1)
                individual_scores += dist

            # 3. FILTRAGE DES OUTLIERS
            # On trie pour identifier les scores qui s'envolent
            sorted_scores, _ = torch.sort(individual_scores)

            # On définit la médiane comme référence de "normalité"
            median_val = torch.median(sorted_scores)

            # Seuil de coupure (ex: 5x la médiane)
            # Toute trajectoire au-dessus de ce seuil est considérée comme un outlier
            cutoff_threshold = median_val * 10.0

            # On ne garde que les scores "raisonnables"
            # On s'assure de garder au moins 50% des données pour éviter de diviser par zéro
            mask = (sorted_scores <= cutoff_threshold)
            filtered_scores = sorted_scores[mask]

            if filtered_scores.numel() == 0:
                # Sécurité au cas où tout serait au-dessus (peu probable)
                final_score = median_val.item()
            else:
                # On retourne la moyenne des scores sains
                # (Plus précis que la médiane seule pour détecter une dérive globale)
                final_score = torch.mean(filtered_scores).item()

        return final_score"""
        
        
        
        
        
        
        
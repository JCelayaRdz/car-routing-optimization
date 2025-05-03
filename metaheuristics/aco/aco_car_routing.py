class ACOCarRouting:
    def __init__(
            self,
            n_ants: int = 10,
            alpha: float = 1,
            beta: float = 5,
            rho: float = 0.8,
            n_cicles_no_improve = 5
        ):
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_cicles_no_improve = 5
        
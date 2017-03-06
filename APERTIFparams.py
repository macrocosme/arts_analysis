class APERTIFparams():
	def __init__(self):
		self.omega = 8.5
		self.G = (26/64.)**2*0.7*0.5 # Scale gain from Parkes
		self.B = 300.
		self.Tsys = 70.
		self.sm = 10.
		self.name = 'APERTIF'
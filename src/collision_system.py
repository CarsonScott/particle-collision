import numpy as np
import pygame

class ParticleSystem:

	def __init__(self, window_size):
		self.dt = 0
		self.masses = []
		self.positions = []
		self.velocities = []
		self.window_size = window_size

	@property
	def size(self):
		return len(self.masses)	

	def mass(self, i):
		return self.masses[i]

	def radius(self, i):
		return self.masses[i] ** 2

	def position(self, i):
		return self.positions[i]

	def velocity(self, i):
		return self.velocities[i]

	def add_particle(self, mass, position=[0,0], velocity=[0,0]):
		self.masses.append(mass)
		self.positions.append(np.array(position))
		self.velocities.append(np.array(velocity))

	def check_collision(self, i, j):
		p1,r1 = self.position(i), self.radius(i)
		p2,r2 = self.position(j), self.radius(j)
		d=np.linalg.norm(p2 - p1)
		return d < r1 + r2

	def compute_velocities(self, i, j):
		p1,v1,m1,r1 = self.position(i), self.velocity(i), self.mass(i), self.radius(i)
		p2,v2,m2,r2 = self.position(j), self.velocity(j), self.mass(j), self.radius(j)

		M = m1 + m2
		n = np.linalg.norm(p1 - p2) ** 2

		u1 = v1 - 2 * m2 / M * np.dot(v1 - v2, p1 - p2) / n * (p1 - p2)
		u2 = v2 - 2 * m1 / M * np.dot(v2 - v1, p2 - p1) / n * (p2 - p1)

		self.velocities[i] = u1
		self.velocities[j] = u2

	def compute_positions(self, i, j):
		p1,v1,m1,r1 = self.position(i), self.velocity(i), self.mass(i), self.radius(i)
		p2,v2,m2,r2 = self.position(j), self.velocity(j), self.mass(j), self.radius(j)

		p1 = p1
		p2 = p2

		f1 = np.linalg.norm(v1)
		f2 = np.linalg.norm(v2)
		F = f1 + f2

		d = np.linalg.norm((p2 + v2 * self.dt) - (p1 + v1 * self.dt))
		e = p2 - p1

		a1 = np.arctan2(e[1], e[0])
		a2 = a1 + np.pi

		if d < r1 + r2:
			n1 = p1 + np.array((np.cos(a2), np.sin(a2))) * (r1 + r2 - d) * 2 * f1 / F
			n2 = p2 + np.array((np.cos(a1), np.sin(a1))) * (r1 + r2 - d) * 2 * f2 / F
			self.positions[i] = n1
			self.positions[j] = n2

	def update_position(self, i):
		w,h = self.window_size
		
		p,v,m,r = self.position(i), self.velocity(i), self.mass(i), self.radius(i)
		p = p + v * self.dt

		x,y = p
		dx,dy = v

		if x - r < 0:
			x = r
			dx = -dx

		if x + r > w:
			x = w - r
			dx = -dx

		if y - r < 0:
			y = r
			dy = -dy

		if y + r > h:
			y = h - r
			dy = -dy

		v = np.array((dx,dy))
		p = np.array((x,y))

		self.velocities[i] = v
		self.positions[i] = p

	def compute(self, dt):
		self.dt = dt
		
		for i in range(self.size):
			for j in range(i, self.size):
				if i != j and self.check_collision(i,j):
					self.compute_velocities(i,j)
					self.compute_positions(i,j)
		
		for i in range(self.size):
			self.update_position(i)

	def draw(self, screen):
		for i in range(self.size):
			p,r = self.position(i), self.radius(i)
			pygame.draw.circle(screen, (255, 255, 255), [int(x) for x in p], int(r))


	def run(self):
		screenw,screenh = self.window_size

		screen = pygame.display.set_mode((screenw, screenh))
		clock = pygame.time.Clock()

		system=ParticleSystem((screenw, screenh))

		for i in range(15):
			m = np.random.randint(3, 10)
			
			p = [np.random.uniform(screenw), 
				 np.random.uniform(screenh)]
			
			v = [np.random.uniform(low=-600, high=600), 
				 np.random.uniform(low=-600, high=600)]

			system.add_particle(m,p,v)	

		running = True
		while running:
			dt = clock.tick() / 1000

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False

			screen.fill((0,0,0))
			
			system.compute(dt)
			system.draw(screen)
			
			pygame.display.flip()
		
		pygame.quit()



if __name__=="__main__":
	ParticleSystem(window_size=(1200, 800)).run()

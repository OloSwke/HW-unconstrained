

module maxlike

	using Distributions, Optim, PyPlot, DataFrames, Debug

	"""
    `input(prompt::AbstractString="")`

    Read a string from STDIN. The trailing newline is stripped.

    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm
	# true coeff vector, number of obs, data matrix X (Nxk), response vector y (binary), and a type of parametric distribution; i.e. the standard normal in our case.

	function makeData(n=10000)

		# Define the coefficient vector
		beta = [ 1; 1.5; -0.5 ]

		# Generate the data matrix from normal distributions
		mu = [0.0, 0.0, 0.0]
		C = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
		X = rand(MvNormal(mu,C),n)
		X = X'

		# Compute the binary response vector
		y = zeros(n)

		epsilon = Normal()
		y_star = X*beta + rand(epsilon,n)

		for i in 1:n
			if y_star[i] > 0
				y[i] = 1
			end
		end

		return Dict("beta" => beta, "numobs" => n, "X" => X, "y" => y, "norm" => Normal())

	end


	# log likelihood function at x
	# function loglik(betas::Vector,d::Dict)

	function loglik(betas::Vector,d::Dict)
		phi = cdf(d["norm"],d["X"]*betas)
		return l = dot(log(phi),d["y"]) + dot(log(1-phi),(1-d["y"]))
	end

	# gradient of the likelihood at x
	function grad!(betas::Vector,storage::Vector,d)
		X = d["X"]
		y = d["y"]

		F(t) = 1 / (1+exp(-t))
		G = zeros(n)
		H = zeros(n)

		for i in 1:n

			G[i] = y[i]*F((-X*betas)[i])
			H[i] = (1-y[i])F((X*betas)[i])

		end

		K = G + H

		grad = K'*X
		storage[1] = grad[1]
		storage[2] = grad[2]
		storage[3] = grad[3]

	end

	# hessian of the likelihood at x
	function hessian!(betas::Vector,storage::Matrix,d)
	end


	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
	end

	"""
	standard errors
	"""
	function se(betas::Vector,d::Dict)
		sqrt(diag(inv_observedInfo(betas,d)))
	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(x0=[0.8,1.0,-0.1],meth=:nelder_mead)

		f(x) = loglik(x,makeData())
		optimize(f,x0,method=:nelder_mead)

	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:bfgs)

		f(x) = loglik(x,makeData())
		gra(s,t) = grad!(s,t,makeData())
		optimize(f, gra, x0, method=:gradient_descent,iterations=1000)

	end

	# function that maximizes the log likelihood with the gradient
	# and hessian with a call to `optimize` and returns the result
	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=:newton)
	end

	# function that maximizes the log likelihood with the gradient
	# and computes the standard errors for the estimates
	# should return a dataframe with 3 rows
	# first column should be parameter names
	# second column "Estimates"
	# third column "StandardErrors"
	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=:bfgs)
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the true value.
	function plotLike()
		fig,axes = subplots(1,3,figsize=(10,3))
		m = 1000

		for i in 1:3

			ax = axes[i,1]
			x = linspace(makeData()["beta"][i]-3.0,makeData()["beta"][i]+3.0,m)

			y = zeros(m)

			for j in 1:m

				betas = makeData()["beta"]
				betas[i] = collect(x)[j]
				y[j] = loglik(betas, makeData())

			end

			ax[:plot](x,y, color="blue")
			ax[:set_title]("Variation of parameter $(i)")
			ax[:yaxis][:set_visible](false)

		end

	end




	function runAll()
		plotLike()
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like: $(m1.minimum)")
		println("maximize_like_grad: $(m2.minimum)")
		println("maximize_like_grad_hess: $(m3.minimum)")
		println("maximize_like_grad_se: $m4")
		println("")
		println("running tests:")
		include("test/runtests.jl")
		println("")
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end


end

module PointSpacing

using Interpolations
using StaticArrays
using LinearAlgebra: norm
using Roots

export material_pt, material_pt_velocity

"""
	material_pt_stretch(s_deform::Real)(s_undef::Real)

For the point described by `s_undef`, the arclength in the undeformed state, compute the arclength of the same "material" point in the deformed state in which the total arclength is `s_deform`. `s_undef` is normalized from 0 to 1 over the compliant section, and `s_deform` is normalized with the same factor.
"""
function material_pt_stretch(s_deform::Real)
	sigmoid(x) = tanh(x)
	dsigmoid(x) = sech(x)^2
	end_slope(a) = s_deform * a * dsigmoid(a / 2) / (2 * sigmoid(a / 2))
	a = find_zero(a -> end_slope(a) - 1, 1)
	function(s_undef::Real)
		s_deform * (1 + sigmoid(a * (s_undef - 0.5)) / sigmoid(a / 2)) / 2
	end
end

function arclengths(f, n)
	t = range(0, 1, n + 1)
	x = @. SVector(f(t))
	ds = norm.(x[2:end] - x[1:end-1])
	s = similar(ds, n + 1)
	s[1] = 0
	@views cumsum!(s[2:end], ds)
	(t, s)
end

function equalize_arclength(f; sample)
	let f = SVector ∘ f
		t, s = arclengths(f, sample)
		param = interpolate((s,), t, Gridded(Linear()))
		s[end], param
	end
end

function material_pt(curve; s_undef_end)
	c = SVector ∘ curve
	s_end, param = equalize_arclength(c; sample=1000)
	s_deform = material_pt_stretch(s_end / s_undef_end)
	function(t::Real)
		c(param(clamp(s_deform(t) * s_undef_end, 0, s_end)))
	end
end

function material_pt_velocity(curve, k; s_undef_end, epsilon=1e-5)
	p1 = material_pt(curve(k - epsilon); s_undef_end)
	p2 = material_pt(curve(k + epsilon); s_undef_end)
	function(t::Real)
		(p2(t) - p1(t)) / (2epsilon)
	end
end

function distribute_points(f; n, sample=16n)
	let f = SVector ∘ f
		s_end, param = equalize_arclength(f; sample)
		ts = param(range(0, s_end, n))
		(ts, f.(ts))
	end
end

function arclength(f; sample)
	t, s = arclengths(f, sample)
	s[end]
end

end

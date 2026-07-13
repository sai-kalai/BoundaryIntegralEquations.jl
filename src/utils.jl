

"""
    starfish(t, R=1, a=0.3, w=5)

starfish domain boundary in 2D as used by [ref](https://github.com/bobbielf2/ZetaTrap2D/blob/master/test_lap2d_hyper_bie.m)

parametrization of a Jordan curve that is a domain boundary

t |--> (x, y)

# Arguments
- `t`: [TODO:parameter]
"""
function starfish(t, R=1, a=0.3, w=5)
    z = (R + a * cos.(w * t)) * cis(t) # parametrization of boundary
    return SA[real(z), imag(z)] # TODO: play with static arrays
end


"""
    ball(r, n)

evenly distributed points in a circumference centered at the origin of radius r

# Arguments
- `r`: radius of the circumference
- `n`: number of points
- return: matrix with datapoints
"""
function ball(r, n)
    z = r .* cis.(2pi * (1:n) / n)
    return hcat(real.(z), imag.(z))
end

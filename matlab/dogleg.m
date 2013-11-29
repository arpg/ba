function y = dogleg()
opt = optimset('Display','iter');


[r,J] = evaluate(ones(5,1));

x = ones(5,1);
tr_size = 2;

while 1

    Jm = J(:,1:2);
    Jl = J(:,3:5);

    g = J'*r;
    

    top = norm(Jm'*r)^2 + norm(Jl'*r)^2;
    bottom = norm((r' * Jm) * Jm' + (r' * Jl) * Jl')^2;
    delta_sd = top/bottom * g;
    %delta_sd = ((g'*g)/(g'*J'*J*g))*g;


    if norm(delta_sd) >= tr_size
        delta_dl = tr_size/norm(delta_sd) * delta_sd;
    else 
        delta_gn = (J'*J)\(J'*r);
        if norm(delta_gn) <= tr_size 
            delta_dl = delta_gn;
        else
            % set up  the quadratic required to find beta
            diff = (delta_gn - delta_sd);
            a = norm(diff) ^ 2;
            b = sum(diff' * delta_sd) * 2;
            c = norm(delta_sd) ^ 2 - tr_size^2;
            beta = (-b^2 + sqrt(b^2 - 4*a*c))/(2*a);
            delta_dl = delta_sd + beta*(delta_gn - delta_sd);
        end          
    end

    %stop condition from step size length
    if norm(delta_dl) < 1e-4*norm(x)
        break;
    else
        x_new = x - delta_dl;
        r_new = evaluate(x_new);
        rho = (norm(r)^2 - norm(r_new)^2) / ...
        (norm(r)^2 - 2*(0.5*norm(r)^2 - (J'*r)'*delta_dl + ...
                        0.5*delta_dl'*J'*J*delta_dl));
        if rho > 0
            x = x_new; 
            [r,J] = evaluate(x);
        else 
            i = 1;
        end
        
        fprintf('rho: %.4f , e: %.4f, tr_size: %d\n',  ...
                rho, norm(r)^2, tr_size);
        
        tr_size = update_radius(tr_size, rho);
    end
end

end

function [tr_new] = update_radius(tr_old, rho)
    if rho > 0
        tr_new = tr_old * 2;
    else
        tr_new = tr_old / 2;
    end
end

function [r,J] = evaluate(x)
    %J = zeros(6,5);
    %J(1:5,1:5) = diag(1:5);
    %J(6,1) = -3;
    %r = J * x;
    x1 = sym('x1','real');
    x2 = sym('x2','real');
    x3 = sym('x3','real');
    x4 = sym('x4','real');
    x5 = sym('x5','real');
    r_sym = [x1^2 ; 
             sin(x1) + sin(x2) + cos(x1) + cos(x2);
             x3*x4*x5 + x1^3;
             cos(x1) + sin(x1);
             x2^4 + x3^2 + x1^2;
             3*x4];
     
    x_sym = [x1 x2 x3 x4 x5]';
    J_sym = jacobian(r_sym, x_sym);
    
    r = subs(r_sym, x_sym, x);
    J = subs(J_sym, x_sym, x);
end

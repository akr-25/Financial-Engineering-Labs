% Simpson's Linear : solving the 2 point Boundary value problem (BVP) by
% using the piecewise-linear basis functions and Simpson's rule for the
% numerical quadrature.

x_min = 0;
x_max = 1;
N = 100;
h = (x_max - x_min)/N;
X = x_min:h:x_max;

A = zeros(N+1, N+1);
B = zeros(N+1, N+1);
C = zeros(N+1, N+1);
d = zeros(N+1, 1);

for i = 1:N-1
% For the basis/row of A
    for j = 0:N
    % For the Uj
        for k = 0:N-1
        % For integration interval
            A(i+1,j+1) = A(i+1,j+1) + (h/6) ...
                * (dphi1(x_min, x_max, N, i, X(k+1), 1)*dphi1(x_min, x_max, N, j, X(k+1), 1) ...
                + dphi1(x_min, x_max, N, i, (X(k+1)+X(k+2))/2, 1)*dphi1(x_min, x_max, N, j, (X(k+1)+X(k+2))/2, 1) ...
                + dphi1(x_min, x_max, N, i, X(k+2), 0)*dphi1(x_min, x_max, N, j, X(k+2), 0));

            B(i+1,j+1) = B(i+1,j+1) + (h/6) ...
                * (bb(X(k+1))*phi1(x_min, x_max, N, i, X(k+1))*dphi1(x_min, x_max, N, j, X(k+1), 1) ...
                + bb((X(k+1)+X(k+2))/2)*phi1(x_min, x_max, N, i,(X(k+1)+X(k+2))/2)*dphi1(x_min, x_max, N, j, (X(k+1)+X(k+2))/2, 1) ...
                + bb(X(k+2))*phi1(x_min, x_max, N, i, X(k+2))*dphi1(x_min, x_max, N, j, X(k+2), 0));

            C(i+1,j+1) = C(i+1,j+1) + (h/6) ...
                * (cc(X(k+1))*phi1(x_min, x_max, N, i, X(k+1))*phi1(x_min, x_max, N, j, X(k+1)) ...
                + cc((X(k+1)+X(k+2))/2)*phi1(x_min, x_max, N, i, (X(k+1)+X(k+2))/2)*phi1(x_min, x_max, N, j, (X(k+1)+X(k+2))/2) ...
                + cc(X(k+2))*phi1(x_min, x_max, N, i, X(k+2))*phi1(x_min, x_max, N, j, X(k+2)));

            if j == 0
                d(i+1) = d(i+1) + (h/6) * (dd(X(k+1))*phi1(x_min, x_max, N, i, X(k+1)) ...
                    + dd((X(k+1)+X(k+2))/2)*phi1(x_min, x_max, N, i, (X(k+1)+X(k+2))/2) ...
                    + dd(X(k+2))*phi1(x_min, x_max, N, i, X(k+2)));
            end
        end
    end
end

AA = A + B + C;
AA(1,1) = 1;
AA(N+1, N+1) = 1;

U = AA\d;
figure; plot(X, U', 'r', 'linewidth',1.5); title('Piecewise-linear basis functions and Simpson’s rule');
xlabel('X'); ylabel('Y'); 

% ***************  HELPER FUNCTIONS  ***************
function [y] = bb(x)
	y = 2*x - 3;
end

function [y] = cc(x)
	y = 0;
end

function [y] = dd(x)
	y = 2*x + 1;
end

function [y] = phi1(x_min, x_max, N, i, x)
	deg = 1;
	h = (x_max - x_min)/N;
	x0 = x_min + (i-1)*h;
	x1 = x_min + i*h;
	x2 = x_min + (i+1)*h;

	if i == 0
		y = 1 - eta(deg, x_min, x_max, N, i, x);
	elseif i == N
		y = eta(deg, x_min, x_max, N, i-1, x);
	elseif (x < x1)
		y = eta(deg, x_min, x_max, N, i-1, x);
	else
		y = 1 - eta(deg, x_min, x_max, N, i, x);
	end

	if (x < x0) || (x > x2)
		y = 0;
    end
end

function [y] = dphi1(x_min, x_max, N, i, x, lm)
	deg = 1;
	h = (x_max - x_min)/N;
	x0 = x_min + (i-1)*h;
	x1 = x_min + i*h;
	x2 = x_min + (i+1)*h;

	if lm == 0
		xx = x - h/3;
	else
		xx = x + h/3;
	end

	if i == 0
		y = -deta(deg, x_min, x_max, N, i, xx);
	elseif i == N
		y = deta(deg, x_min, x_max, N, i-1, xx);
	elseif (xx < x1)
		y = deta(deg, x_min, x_max, N, i-1, xx);
	else
		y = -deta(deg, x_min, x_max, N, i, xx);
	end

	if (xx < x0) || (xx > x2)
		y = 0;
    end
end

function [y] = deta(deg, x_min, x_max, N, i, x)
	h = (x_max - x_min)/N;
	x1 = x_min + i*h;
	x2 = x_min + (i+deg)*h;

	if (x >= x1) && (x <= x2)
		y = 1/h;
	else
		y = 0;
    end
end

function [y] = eta(deg, x_min, x_max, N, i, x)
	h = (x_max - x_min)/N;
	x1 = x_min + i*h;
	x2 = x_min + (i+deg)*h;
	if (x >= x1) && (x <= x2)
		y = (x - x1)/(x2 - x1);
	else
		y = 0;
    end
end
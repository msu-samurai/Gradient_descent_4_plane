#include <stdio.h>
#include <iostream>
//#include "Vector.h"
#include <math.h>
#define e2  0.0000001










class Vector
{
	public:
	int n;
	double * elems;

	//Vector (int m); //default constructor
	Vector  (int m):n(m)
	{
		elems = new double[m];
		for (int i = 0; i < n; i ++)
		{
				elems[i] = 0;
		}
	}


	//Vector (int m, double * elements);
	Vector  (int m, double * elements):n(m)
	{
		elems = new double [n];
		for (int i = 0; i < n; i ++)
			elems[i] = elements[i];

	}


	//Vector (const Vector & x); //constructor of copying
	Vector  (const Vector & x)
	{
		n = x.n;
		elems = new double[n];
		for (int i = 0; i < n; i ++)
			elems[i] = x.elems[i];
	}

	//~Vector ();
	~Vector ()
	{
		delete [] elems;
	}
	//funcs and operators

	//Vector & operator = (const Vector & x);
	Vector &  operator = (const Vector & x)
	{
		if(elems != NULL)
			delete [] elems;
		n = x.n;
		elems = new double[n];
		for (int i = 0; i < n; i ++)
			elems[i] = x.elems[i];
		return * this;
	}

	//Vector & operator *= (const Matrix & x);
	/*Vector & operator *= (const Matrix & x)
	{

		if (n != x.n)
		{
			std:: cout << "Bad, nothing changed\n";
			return * this;
		}
		
		double elem, * elements;
		elements = new double [n];
		for (int i = 0; i < n; i ++)
		{
			elem = 0;
			for (int k = 0; k < n; k ++)
				elem += elems[k] * x.elems[i * n + k];
			elements[i] = elem;
		}
		for (int i = 0; i < n; i ++)
			elems[i] = elements [i];
		delete [] elements;
		return * this;

	}*/
	//Vector & operator *= (const double a);
	Vector & operator *= (const double a)
	{
		for (int i = 0; i < n; i ++)
			elems[i] *= a;
		return * this;
	}

	Vector operator - (const Vector & x)
	{
		Vector res(x.n);
		for (int j = 0; j < x.n; j ++)
		{
			res.elems[j] = elems[j] - x.elems[j];
		}
		return res;
	}

	Vector operator * (const double a)
	{
		Vector res(n);
		for (int j = 0; j < n; j ++)
		{
			res.elems[j] = elems[j] * a;
		}
		return res;
	}
	
	//Vector & operator += (const double a);
	Vector & operator += (const double a)
	{
		for (int i = 0; i < n; i ++)
			elems[i] += a;
		return * this;
	}


	//Vector & operator -= (const double a);
	Vector &  operator -= (const double a)
	{
		for (int i = 0; i < n; i ++)
			elems[i] -= a;
		return * this;
	}

	//Vector scalar(const Vector & y);
	Vector vector_product_aka_scalar (const Vector & y)
	{
		Vector result(y.n);
		for (int j = 0; j < y.n; j ++)
			result.elems[j] = elems[j] * y.elems[j];
		return result;
	}

	double scalar_product (const Vector & y)
	{
		double result = 0;
		for (int j = 0; j < y.n; j ++)
			result += elems[j] * y.elems[j];
		return result;
	}

	double norm()
	{
		return sqrt(scalar_product(* this));
	}

	//Vector operator +(const Vector & x);
	Vector operator +(const Vector & x)
	{
		Vector result(x.n);
		for (int j = 0; j < x.n; j ++)
			result.elems[j] = elems[j] + x.elems[j];
		return result;
	}

	/*friend Vector operator * (const Matrix & A,const Vector & b)
	{
		if (A.n != b.n)
		{
			std:: cout << "Bad, nothing changed\n";
			return b;
		}
		int m = b.n;
		double elem, * elements;
		elements = new double [m];
		for (int i = 0; i < m; i ++)
		{
			elem = 0;
			for (int k = 0; k < m; k ++)
				elem += b.elems[k] * A.elems[i * m + k];
			elements[i] = elem;
		}
		for (int i = 0; i < m; i ++)
			b.elems[i] = elements [i];
		delete [] elements;
		return b;
	}

	friend Vector & VSolveLin(Vector & x, Matrix & A, const Vector & b)
	{
		if (A.n != b.n)
		{
			std::cout << "Wrong sizes\n";
			return x;
		}
		if (fabs(A.Det()) < (1e-14))
		{
			std::cout << "NULL solution. May have other solutions\n";
			Vector c(x.n);
			x = c;
			return x;
		}
		Matrix C(x.n);
		Inverse(C,A);
		x = C * b;
		return x;


	}
*/
};







	

	

	
	

	
	
	

	

	

	


	







struct R3_point
{
	double x;
	double y;
	double z;
};




double function(double t);
double function(double t)
{
	return cos(t);
}

double n_function(Vector t, const struct R3_point * points, const int quantity);
double n_function(Vector t, const struct R3_point * points, const int quantity)
{
	Vector G(quantity);
	for (int i = 0; i < quantity; i ++)
	{
		G.elems[i] = (fabs(t.elems[0]*(points[i].x) + t.elems[1]*(points[i].y) + t.elems[2]*(points[i].z)  + t.elems[3]))/(sqrt( t.elems[0]*t.elems[0] + t.elems[1]*t.elems[1] + t.elems[2]*t.elems[2] ));
	}
	//G.elems[0] = 3 * t.elems[0] - cos(t.elems[1] * t.elems[2]) - 1.5;
	//G.elems[1] = 4 * t.elems[0] * t.elems[0] - 625 * t.elems[1] * t.elems[1] + 2 * t.elems[1] - 1;
	//G.elems[2] = exp(-1 * t.elems[0] * t.elems[1]) + 20*t.elems[2] + (10 * 3.1415 - 3)/3;
	return sqrt( G.scalar_product(G));
	//return (1 - t.elems[0]) * (1 - t.elems[0]) + 100 * (t.elems[1] - t.elems[0] * t.elems[0]) * (t.elems[1] - t.elems[0] * t.elems[0]);
	//return sin(0.5 * (t.elems[0])*(t.elems[0]) - (1/4) * (t.elems[1])*(t.elems[1]) + 3) * cos(2 * t.elems[0] + 1 + exp(t.elems[1]));
	//return (t.elems[0])*(t.elems[0]) + (t.elems[1])*(t.elems[1]);
}

/*Vector vector_3_function(Vector t);
Vector vector_3_function(Vector t)
{
	Vector result(3);
	result.elems[0] = 3 * t.elems[0] - cos(t.elems[1] * t.elems[2]) - 1.5;
	result.elems[1] = 4 * t.elems[0] * t.elems[0] - 625 * t.elems[1] * t.elems[1] + 2 * t.elems[1] - 1;
	result.elems[2] = exp(-1 * t.elems[0] * t.elems[1]) + 20*t.elems[2] + (10 * 3.1415 - 3)/3;
	return result;
}

double n_function_with_function(Vector t, Vector (*f)(Vector));
double n_function_with_function(Vector t, Vector (*f)(Vector))
{
	return 0.5
}*/

double grad(double (*f)(double), double x);
double grad(double (*f)(double), double x)
{
	return (f(x + e2) - f(x))/(e2);
}

Vector n_grad(double (*f)(Vector, const struct R3_point *, const int ), Vector & x, int dim, const struct R3_point * points, const int quantity);
Vector n_grad(double (*f)(Vector, const struct R3_point *, const int), Vector & x, int dim, const struct R3_point * points, const int quantity)
{
	Vector grad(dim), eps(dim), x_i(dim);

	for (int i = 0; i < dim; i ++)
	{
		for(int j = 0; j < dim; j ++)
		{
			if (i != j)
				 eps.elems[j] = 0;
			else
				eps.elems[j] = 1;
		}


		//x_i = x.scalar(eps);
		eps *= e2;

		
		grad.elems[i] = (f(x + eps, points, quantity) - f(x, points, quantity))/(e2);
		
	}
	return grad;

	
}

double grad_descent(double (*f)(double), double x_curr);
double grad_descent(double (*f)(double), double x_curr)
{	
	int itteration = 0;
	double x_next = x_curr - 0.001 * grad(f, x_curr);
	std::cout << "Itteration: " << itteration << "\n";
	while((fabs(x_next - x_curr) > e2)  || (fabs(f(x_next) - f(x_curr)) > e2) || (grad(f,x_curr) > e2))
	{
		itteration ++;
		x_curr = x_next;
		x_next = x_curr - 0.001 * grad(f, x_curr);
		std::cout << "Itteration: " << itteration << "\n";
	}
	return x_next;
	
}

Vector n_grad_descent(double (*f)(Vector, const struct R3_point *, const int ), Vector & x_curr , int dim, const struct R3_point * points, const int quantity);
Vector n_grad_descent(double (*f)(Vector, const struct R3_point *, const int), Vector & x_curr , int dim, const struct R3_point * points, const int quantity)
{
	int itteration = 0;
	Vector x_next(dim);
	Vector gradient(dim);
	double velocity = 0.001;


	gradient = n_grad(f, x_curr, dim, points, quantity);
	

	x_next = x_curr - gradient * velocity;
	//std::cout << "Itteration: " << itteration << "\n";
	while(((x_next - x_curr).norm() > e2)  || (fabs(f(x_next, points, quantity) - f(x_curr, points, quantity)) > e2) || ((gradient).norm() > e2))
	{
		itteration ++;
		velocity = (fabs(  (x_next - x_curr).scalar_product( (n_grad(f, x_next, dim, points, quantity)) - (gradient) ) ))/( (((n_grad(f, x_next, dim, points, quantity)) - (gradient)).norm()) * (((n_grad(f, x_next, dim, points, quantity)) - (gradient)).norm()) );


		x_curr = x_next;


		gradient = n_grad(f, x_curr, dim, points, quantity);
		x_next = x_curr -  gradient * velocity;
		//std::cout << "Itteration: " << itteration << "\n";
	}
	return x_next;
}



int main()
{
	//int dim;
	double vector[4] = {0.1,1, 0.1, 0};
	int quantity = 5;
	double buf;
	struct R3_point * points;
	points = (struct R3_point *)malloc(quantity * sizeof(struct R3_point));
	for (int i = 0; i < quantity - 1; i ++)
	{
		points[i].x = i;
		points[i].y = i + 1;
		points[i].z = i + 2;
	}

	points[4].x = 0;
	points[4].y = 8;
	points[4].z = 9;
	 
	//std::cout << "Enter dimension of preimage: ";
	//std:: cin >> dim;
	Vector plane(4, vector);
	/*double res = grad_descent(function, 0.5);
	std::cout << res << "\n" << function(res);*/


	Vector res = n_grad_descent(n_function, plane, 4, points, quantity);
	
	 
	for (int i = 0; i < res.n; i ++)
		std::cout << res.elems[i] << "\n";

	if (res.elems[0] > 1e-4)
	{
		buf = res.elems[0];
		for (int i = 0; i < res.n; i ++)
			res.elems[i] /= buf;
	}
	else if (res.elems[1] > 1e-4)
	{
		buf = res.elems[1];
		for (int i = 1; i < res.n; i ++)
			res.elems[i] /= buf;
	}
	else if (res.elems[2] > 1e-4)
	{
		buf = res.elems[2];
		for (int i = 2; i < res.n; i ++)
			res.elems[i] /= buf;
	}

	std::cout << "after division" << std::endl;

	for (int i = 0; i < res.n; i ++)
		std::cout << res.elems[i] << "\n";

	std::cout << "========\n"<< n_function(res, points, quantity);


	return 0;
}
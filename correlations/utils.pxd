cdef public api:
    cpdef double integrand(double k, double phi, double theta, int ikx, int iky, int ikz, int ikk,
                           double dx, double dy, double dz, double R1, double R2)

    cpdef double integrand_lambdaCDM(double phi, double theta, int ikx, int iky, int ikz, int ikk,
                                     double dx, double dy, double dz, double R1, double R2)

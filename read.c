#include <stdio.h>
#include <stdlib.h>

int main(void)
{
	FILE *f = fopen("eq.txt", "r");
	if ( f == NULL)
		printf("ERROR");
	int x[10];
	for (int i = 0; i< 10 ;i++)
	{
		fscanf(f, "%d,", &x[i]);
		printf("%d\n", x[i]);
	}


	fclose(f);

	return 0;
}

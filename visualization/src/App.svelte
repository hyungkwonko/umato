<script lang="ts">
	import Scatterplot from "./Scatterplot.svelte";
	import Scatterplot2 from "./Scatterplot2.svelte";
	import Scatterplot3 from "./Scatterplot3.svelte";
	import Scatterplot4 from "./Scatterplot4.svelte";
	import Scatterplot5 from "./Scatterplot5.svelte";
	import Scatterplot6 from "./Scatterplot6.svelte";
	import Scatterplot7 from "./Scatterplot7.svelte";
	import Linechart from "./Linechart.svelte";
	import Arrow from "./Arrow.svelte";
	import Head from "./Head.svelte";
	let dnames = ["mnist", "fmnist", "kmnist"];
	// let dnames = ["spheres"];
	// let dnames = ["spheres", "mnist", "fmnist", "kmnist"];
	let dnames2 = ["scurve", "swissroll"];
	let type = 9;

</script>

<main>
	{#if type == 0}
		<table>
			<tr>
				<td><Scatterplot3 dname={"classification"} algoname={"umato"}></Scatterplot3></td>
			</tr>
		</table>
	{:else if type == 1}
		<table>
			<tr>
				<td><Scatterplot dname={"multiphase"} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={"multiphase"} algoname={"umato_3-phases"}></Scatterplot></td>
				<td><Scatterplot dname={"multiphase"} algoname={"umato_4-phases"}></Scatterplot></td>
			</tr>
		</table>
	{:else if type == 2}
		{#each dnames as dname}
			<!-- <h1>{dname.charAt(0).toUpperCase() + dname.slice(1)} data</h1> -->
			<table>
				<tr>
					<td><Head dname={dname}></Head></td>
					<td><Scatterplot dname={dname} algoname={"pca"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"isomap"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"tsne"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"umap"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"topoae"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"atsne"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"umato"}></Scatterplot></td>
				</tr>
			</table>
		{/each}
	{:else if type == 3}
		<table>
			<tr>
				<td><Linechart></Linechart></td>
			</tr>
		</table>
	{:else if type == 4}
		<table>
			<tr>
				<td><Scatterplot dname={"local"} algoname={"MNIST_LR=0.1"}></Scatterplot></td>
				<td><Scatterplot dname={"local"} algoname={"FMNIST_LR=0.1"}></Scatterplot></td>
				<td><Scatterplot dname={"local"} algoname={"KMNIST_LR=0.1"}></Scatterplot></td>
			</tr>
		</table>
	{:else if type == 5}
		<table>
			<tr>
				<td><Scatterplot dname={dnames[0]} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[1]} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[2]} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[3]} algoname={"umato"}></Scatterplot></td>
			</tr>
		</table>
	{:else if type == 6}
		{#each dnames2 as dname}
			<!-- <h1>{dname.charAt(0).toUpperCase() + dname.slice(1)} data</h1> -->
			<table>
				<tr>
					<td><Head dname={dname}></Head></td>
					<td><Scatterplot4 dname={dname} algoname={"pca"}></Scatterplot4></td>
					<td><Scatterplot4 dname={dname} algoname={"isomap"}></Scatterplot4></td>
					<td><Scatterplot4 dname={dname} algoname={"tsne"}></Scatterplot4></td>
					<td><Scatterplot4 dname={dname} algoname={"umap"}></Scatterplot4></td>
					<td><Scatterplot4 dname={dname} algoname={"umato"}></Scatterplot4></td>
				</tr>
			</table>
		{/each}
	{:else if type == 7}
		<table>
			<tr>
				<td><Scatterplot dname={'spheres'} algoname={"class-wise separation"}></Scatterplot></td>
				<td><Scatterplot dname={'spheres'} algoname={"PCA_"}></Scatterplot></td>
				<td><Scatterplot dname={'spheres'} algoname={"Random"}></Scatterplot></td>
				<td><Scatterplot dname={'spheres'} algoname={"Spectral embedding"}></Scatterplot></td>
			</tr>
		</table>
	{:else if type == 8}
		<table>
			<tr>
				<td><Scatterplot5 dname={'run'} algoname={'umap_200'} x={-7} y={17}></Scatterplot5></td>
				<td><Scatterplot5 dname={'run'} algoname={'umap_1000'} x={-7} y={17}></Scatterplot5></td>
				<td><Scatterplot5 dname={'run'} algoname={'umap_5000'} x={-7} y={17}></Scatterplot5></td>
			</tr>
		</table>
		<table>
			<tr>
				<td><Scatterplot5 dname={'run'} algoname={'umato_200'} x={0} y={10}></Scatterplot5></td>
				<td><Scatterplot5 dname={'run'} algoname={'umato_1000'} x={0} y={10}></Scatterplot5></td>
				<td><Scatterplot5 dname={'run'} algoname={'umato_5000'} x={0} y={10}></Scatterplot5></td>
			</tr>
		</table>
	{:else if type == 9}
		<table>
			<tr>
				<td><Scatterplot6 dname={'allen'} algoname={'tsne'}></Scatterplot6></td>
				<!-- <td><Scatterplot6 dname={'allen'} algoname={'UMAP_200_epochs'}></Scatterplot6></td>
				<td><Scatterplot6 dname={'allen'} algoname={'UMAP_2000_epochs'}></Scatterplot6></td> -->
				<td><Scatterplot7 dname={'allen'} algoname={'UMAP_200_epochs'} x={-15} y={25}></Scatterplot7></td>
				<td><Scatterplot7 dname={'allen'} algoname={'UMAP_2000_epochs'} x={-15} y={25}></Scatterplot7></td>
			</tr>
		</table>
		<br>
		<table>
			<tr>
				<td><Scatterplot6 dname={'allen'} algoname={'UMATO_100_epochs'}></Scatterplot6></td>
				<td><Scatterplot6 dname={'allen'} algoname={'UMATO_1000_epochs'}></Scatterplot6></td>
			</tr>
		</table>
	{:else}
		<!-- <h1>UMATO pipeline</h1> -->
		<table>
			<tr>
				<td><Scatterplot2 filename={"(A)"} mystring={"(A) Hub points_initialization"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 filename={"(B)"} mystring={"(B) After optimizing_positions of hubs points"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 filename={"(C)"} mystring={"(C) After embedding_expanded nearest neighbors"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 filename={"(D)"} mystring={"(D) Final result"}></Scatterplot2></td>
			</tr>
		</table>
	{/if}
</main>
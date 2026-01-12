    
    # Step 6: Convert to ArterialGNet format
    print("\n[Step 5] Converting to ArterialGNet format...")
    gnet_result = centerline_to_arterial_gnet_graph(
        results['stage7']['graph'],
        output_pickle_path=str(output_dir / 'arterial_gnet_graph.pkl'),
        include_segment_graph=True,
    )
    print(f"  Dense graph: {gnet_result['dense_graph'].number_of_nodes()} nodes")
    print(f"  Segment graph: {gnet_result['segment_graph'].number_of_nodes()} segments")
    
    # Step 7: Visualize
    print("\n[Step 6] Creating visualizations...")
    visualize_results(results['stage7']['graph'], vessel_mask, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - realistic_vessel_mask.nii.gz (input)")
    print(f"  - centerline/centerline.pkl (NetworkX graph)")
    print(f"  - centerline/centerline_nodes.json")
    print(f"  - centerline/centerline_edges.json")
    print(f"  - evc_graph.pkl (EVC format)")
    print(f"  - arterial_gnet_graph.pkl (ArterialGNet format)")
    print(f"  - morphology_features.json")
    print(f"  - centerline_visualization.png")
    print(f"  - centerline/pipeline.log")
    
    print("\nNext steps:")
    print("  1. View mask: fsleyes realistic_vessel_mask.nii.gz")
    print("  2. Use EVC for vessel classification")
    print("  3. Use ArterialGNet for vessel labeling")
    print("  4. Extract features for stroke prediction modeling")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

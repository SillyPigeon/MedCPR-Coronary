import argparse
from cpr import extract_coordinates, curve_planar_reformat


def main():
    parser = argparse.ArgumentParser(
        description="Curved Planar Reformation (CPR) Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py extract -s vessel_seg.nii.gz -o points.npy
  python main.py cpr -i ct_scan.nii.gz -o cpr.nii.gz -f 50
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Subcommand: extract coordinates
    extract_parser = subparsers.add_parser('extract', help='Extract centerline coordinates')
    extract_parser.add_argument('--seg', '-s', required=True, help='Segmentation file')
    extract_parser.add_argument('--output', '-o', default=None, help='Output points file (.npy) (default: points.npy)')
    extract_parser.add_argument('--skip', '-k', type=int, default=10, help='Skip factor for downsampling (default: 10)')
    
    # Subcommand: CPR
    cpr_parser = subparsers.add_parser('cpr', help='Perform curved planar reformation')
    cpr_parser.add_argument('--input', '-i', required=True, help='Input image file')
    cpr_parser.add_argument('--points', '-p', required=True, help='Input points file (.npy)')
    cpr_parser.add_argument('--output', '-o', default=None, help='Output CPR image (.nii.gz) (default: reformatted_image.nii.gz)')
    cpr_parser.add_argument('--fov', '-f', type=int, required=True, help='Field of view (mm)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'extract':
        print(f"Extracting coordinates from {args.seg}...")
        result = extract_coordinates(args.seg, args.output, skip=args.skip)
        if result is False:
            print("Error: Failed to extract coordinates.")
        else:
            print(f"Successfully extracted {len(result)} points to {args.output}")
    
    elif args.command == 'cpr':
        curve_planar_reformat(args.input, args.points, args.output, args.fov)


if __name__ == '__main__':
    main()



